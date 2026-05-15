"""
Phase C (C2-only): DeepSeek-V4-Flash description-only rederivation.

Input:
  --stage-b-jsonl : Stage B output, one record per id with scene_description.
  --stage-a-jsonl : Stage A output (16 rows per id), source of question + gt_answer.

Per id (where Stage B parse_ok_b2=True and scene_description non-empty):
  C2 call: prompt = (question, scene_description), no gold, no trace.
  Harvest <think> => c2_think; <answer>...</answer> block => c2_answer.
  Deterministic c2_lands_on_gold via _norm-based match against gt_answer.

Output schema (per id):
  {id, c2_think, c2_answer, c2_lands_on_gold, raw_c2,
   parse_ok_c2, oversized_c2, c2_chat_error}

Resume-safe (output IS the source of truth). Single-instance via fcntl.flock.
"""
import argparse, fcntl, json, re, sys, time
from pathlib import Path

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
THINK_RE  = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE    = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS_RE.sub(" ", _PUNCT_RE.sub("", (s or "").lower())).strip()


def compute_ac(candidate: str, gold: str) -> int:
    c, g = _norm(candidate), _norm(gold)
    if not c or not g:
        return 0
    return int(c == g or g in c or c in g)


def load_processed_ids(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    ids = set()
    with out_path.open() as f:
        for line in f:
            try:
                ids.add(json.loads(line)["id"])
            except Exception:
                pass
    return ids


def build_qa_index(stage_a_path: Path) -> dict:
    """Map id -> {question, gt_answer}. Reads first sample per id only."""
    out = {}
    with stage_a_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            id_ = r.get("id")
            if not id_ or id_ in out:
                continue
            out[id_] = {
                "question": r.get("prompt") or r.get("question") or "",
                "gt_answer": r.get("gt_answer") or r.get("gold_answer") or "",
            }
    return out


def stream_stage_b(stage_b_path: Path):
    """Yield (id, scene_description) for usable Stage B records."""
    with stage_b_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("parse_ok_b2"):
                continue
            if r.get("all_b1_failed"):
                continue
            sd = (r.get("scene_description") or "").strip()
            if not sd:
                continue
            yield r["id"], sd


def render_c2(template: str, question: str, scene_description: str) -> str:
    return (template
            .replace("{question}", str(question))
            .replace("{scene_description}", str(scene_description)))


def parse_c2(text: str):
    """Returns (c2_think, c2_answer, parse_ok).
    c2_think prefers <think>...</think>; falls back to text before <answer>."""
    if not text:
        return "", "", False
    ans_m = ANSWER_RE.search(text)
    if not ans_m:
        return "", "", False
    c2_answer = ans_m.group(1).strip()
    think_m = THINK_RE.search(text)
    if think_m:
        c2_think = think_m.group(1).strip()
    else:
        c2_think = text[:ans_m.start()].strip()
    return c2_think, c2_answer, bool(c2_answer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-b-jsonl", required=True)
    ap.add_argument("--stage-a-jsonl", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="ids per llm.chat call")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-output-tokens", type=int, default=8192)
    ap.add_argument("--prompt-token-safety", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--enforce-eager", action="store_true", default=True)
    args = ap.parse_args()

    stage_b = Path(args.stage_b_jsonl)
    stage_a = Path(args.stage_a_jsonl)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    template = Path(args.template).read_text()

    out_f = out_path.open("a")
    try:
        fcntl.flock(out_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"another deepseek_polish already holds {out_path}")

    processed = load_processed_ids(out_path)
    print(f"[boot] {len(processed)} ids already polished in {out_path}", flush=True)
    print(f"[boot] indexing Stage A QAs from {stage_a}", flush=True)
    t0 = time.time()
    qa_idx = build_qa_index(stage_a)
    print(f"[boot] QA index: {len(qa_idx)} ids in {time.time()-t0:.1f}s", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        kv_cache_dtype="fp8",
    )
    sampling = SamplingParams(
        temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_output_tokens,
    )
    tok = llm.get_tokenizer()
    prompt_budget = args.max_model_len - args.max_output_tokens - args.prompt_token_safety

    def _msg_token_len(msg):
        text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        return len(tok.encode(text))

    # Build the queue of (id, prompt_text)
    queue = []
    skipped_no_qa = 0
    for id_, sd in stream_stage_b(stage_b):
        if id_ in processed:
            continue
        qa = qa_idx.get(id_)
        if not qa:
            skipped_no_qa += 1
            continue
        prompt = render_c2(template, qa["question"], sd)
        queue.append((id_, prompt, qa["gt_answer"]))
        if args.limit and len(queue) >= args.limit:
            break
    print(f"[boot] queue={len(queue)} ids to polish "
          f"(skipped_no_qa={skipped_no_qa})", flush=True)

    total_done = 0
    for batch_start in range(0, len(queue), args.batch_size):
        batch = queue[batch_start: batch_start + args.batch_size]

        messages, metas, oversized = [], [], []
        for id_, prompt, gold in batch:
            msg = [{"role": "user", "content": prompt}]
            try:
                plen = _msg_token_len(msg)
            except Exception as e:
                print(f"[WARN] tokenize failed id={id_}: {e}", flush=True)
                plen = prompt_budget + 1
            if plen > prompt_budget:
                # Oversize: emit failure record straight away.
                rec = {
                    "id": id_,
                    "c2_think": "",
                    "c2_answer": "",
                    "c2_lands_on_gold": False,
                    "raw_c2": "",
                    "parse_ok_c2": False,
                    "oversized_c2": True,
                    "c2_chat_error": "",
                }
                out_f.write(json.dumps(rec) + "\n")
                oversized.append(id_)
                continue
            messages.append(msg)
            metas.append((id_, gold))

        if oversized:
            out_f.flush()
            print(f"[batch {batch_start}] oversized={len(oversized)}", flush=True)

        if not messages:
            continue

        try:
            outputs = llm.chat(messages, sampling)
        except Exception as e:
            err = f"chat_error: {type(e).__name__}: {e}"
            print(f"[WARN] {err}", flush=True)
            for id_, gold in metas:
                rec = {
                    "id": id_,
                    "c2_think": "", "c2_answer": "",
                    "c2_lands_on_gold": False, "raw_c2": "",
                    "parse_ok_c2": False, "oversized_c2": False,
                    "c2_chat_error": err,
                }
                out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            continue

        for (id_, gold), out in zip(metas, outputs):
            raw = out.outputs[0].text if out.outputs else ""
            c2_think, c2_answer, parse_ok = parse_c2(raw)
            lands = bool(compute_ac(c2_answer, gold)) if parse_ok else False
            rec = {
                "id": id_,
                "c2_think": c2_think,
                "c2_answer": c2_answer,
                "c2_lands_on_gold": lands,
                "raw_c2": raw,
                "parse_ok_c2": parse_ok,
                "oversized_c2": False,
                "c2_chat_error": "",
            }
            out_f.write(json.dumps(rec) + "\n")
        out_f.flush()

        total_done += len(metas)
        print(f"[batch {batch_start}] processed={len(metas)}  total_done={total_done}",
              flush=True)

    print(f"[done] total_done={total_done}", flush=True)


if __name__ == "__main__":
    main()
