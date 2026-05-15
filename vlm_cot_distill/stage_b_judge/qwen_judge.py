"""
Stage B: Qwen3-VL-235B-Thinking-FP8 judge + scene description.

For each Stage A id (with 16 candidates):
  B1: score each candidate individually (image + Q + gold + grounding_i + trace_i + answer_i)
       => 16 calls per id, 4-dim score vector each
  Pick winner: argmax(score_total) where total = halluc + visual_grounding + reasoning + 5*answer_correctness
               (tie-break: lowest sample_idx)
  B2: scene description (image + Q + gold + winning grounding/trace/answer)
       => 1 call per id; native <think> stripped from output

If all 16 B1 calls fail to parse, default winner = sample_idx 0 with
best_idx=0 and all_b1_failed=True; B2 still runs.

Race-safe consumer of Stage A's actively-growing JSONL (same contract as
the prior teacher_enrich.py):
- append-only reads, parse-error on trailing line => wait
- id is "complete" iff 16 samples present AND a strictly-later id seen
- output file IS the source of truth for resume (no sidecar)
- single-instance enforced via fcntl.flock on output file
"""
import argparse, fcntl, json, re, sys, time
from pathlib import Path
from collections import defaultdict
from PIL import Image

N_SAMPLES = 16
JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
BARE_JSON_RE  = re.compile(r"(\{[^{}]*\"hallucination\"[^{}]*\})", re.DOTALL)


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


def scan_complete_ids(input_path: Path, processed: set):
    """List of (id, [16 sample dicts]) for ids that are complete and unprocessed."""
    by_id = defaultdict(dict)
    order = []
    with input_path.open() as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.endswith("\n"):
                break  # in-flight tail line — wait
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                raise
            id_ = r["id"]
            if id_ not in by_id:
                order.append(id_)
            by_id[id_][r["sample_idx"]] = r
    if len(order) < 2:
        return []
    last_id = order[-1]
    out = []
    for id_ in order:
        if id_ == last_id:
            continue
        if id_ in processed:
            continue
        samples = by_id[id_]
        if len(samples) != N_SAMPLES:
            continue
        out.append((id_, [samples[i] for i in range(N_SAMPLES)]))
    return out


def render_b1(template, sample, gold, question):
    return (template
            .replace("{question}", str(question))
            .replace("{gold_answer}", str(gold))
            .replace("{grounding}", (sample.get("grounding") or "").strip())
            .replace("{trace}", (sample.get("thinking") or sample.get("raw") or "").strip())
            .replace("{candidate_answer}", (sample.get("answer") or "").strip()))


def render_b2(template, winner, gold, question):
    return (template
            .replace("{question}", str(question))
            .replace("{gold_answer}", str(gold))
            .replace("{grounding}", (winner.get("grounding") or "").strip())
            .replace("{trace}", (winner.get("thinking") or "").strip())
            .replace("{answer}", (winner.get("answer") or "").strip()))


def parse_b1(text):
    """Reject if any score is missing or out of [1,5]. Accept fenced ```json
    block OR a bare {...} containing 'hallucination'. Coerce digit-strings
    (e.g. "4") to int. answer_correctness is computed deterministically
    post-hoc, NOT from the judge."""
    if not text:
        return None
    m = JSON_BLOCK_RE.search(text) or BARE_JSON_RE.search(text)
    if not m:
        return None
    try:
        d = json.loads(m.group(1))
    except Exception:
        return None
    for dim in ("hallucination", "visual_grounding", "reasoning_quality"):
        sub = d.get(dim)
        if not isinstance(sub, dict):
            return None
        s = sub.get("score")
        if isinstance(s, str) and s.strip().isdigit():
            s = int(s.strip())
            sub["score"] = s
        if not isinstance(s, int) or not (1 <= s <= 5):
            return None
    return d


_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")

def _norm(s: str) -> str:
    return _WS_RE.sub(" ", _PUNCT_RE.sub("", (s or "").lower())).strip()

def compute_ac(candidate_answer: str, gold_answer: str) -> int:
    """Deterministic answer_correctness via case/punct-tolerant string match.
    Match if normalized strings are equal OR one contains the other (handles
    'the X' vs 'X' and 'I think it's X' vs 'X')."""
    c, g = _norm(candidate_answer), _norm(gold_answer)
    if not c or not g:
        return 0
    return int(c == g or g in c or c in g)


def score_total(d):
    return (d["hallucination"]["score"]
            + d["visual_grounding"]["score"]
            + d["reasoning_quality"]["score"]
            + 5 * d["answer_correctness"])


def strip_native_think(text: str) -> str:
    """B2 returns <think>...</think><description>. Discard everything up to
    and including the first </think>. If no </think> found, return as-is."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return re.sub(r"^\s*thinking\s*", "", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--b1-template", required=True)
    ap.add_argument("--b2-template", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, process at most N ids and exit (test mode)")
    ap.add_argument("--watch", action="store_true",
                    help="Loop forever, sleeping when no new complete ids")
    ap.add_argument("--watch-sleep", type=int, default=60)
    ap.add_argument("--ids-per-batch", type=int, default=4,
                    help="ids per llm.chat call (each adds 16 B1 prompts)")
    ap.add_argument("--max-model-len", type=int, default=24576)
    ap.add_argument("--max-output-tokens", type=int, default=8192,
                    help="Default for both B1 and B2 if -b1/-b2 not set")
    ap.add_argument("--max-output-tokens-b1", type=int, default=None)
    ap.add_argument("--max-output-tokens-b2", type=int, default=None)
    ap.add_argument("--prompt-token-safety", type=int, default=256,
                    help="Reserve this many tokens beyond max_output as headroom in oversize check")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--enforce-eager", action="store_true", default=True)
    ap.add_argument("--no-think-b1", action="store_true",
                    help="Disable native <think> on B1 calls (chat_template_kwargs enable_thinking=False)")
    ap.add_argument("--no-think-b2", action="store_true",
                    help="Disable native <think> on B2 calls")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    b1_template = Path(args.b1_template).read_text()
    b2_template = Path(args.b2_template).read_text()

    out_f = out_path.open("a")
    try:
        fcntl.flock(out_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"another qwen_judge already holds {out_path}")

    processed = load_processed_ids(out_path)
    print(f"[boot] {len(processed)} ids already enriched in {out_path}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        limit_mm_per_prompt={"image": 1},
    )
    max_b1 = args.max_output_tokens_b1 or args.max_output_tokens
    max_b2 = args.max_output_tokens_b2 or args.max_output_tokens
    sampling_b1 = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        max_tokens=max_b1,
    )
    sampling_b2 = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        max_tokens=max_b2,
    )
    tok = llm.get_tokenizer()
    b1_budget = args.max_model_len - max_b1 - args.prompt_token_safety
    b2_budget = args.max_model_len - max_b2 - args.prompt_token_safety

    def _msg_token_len(msg):
        # msg is the inner list-of-dicts (one chat); apply_chat_template needs the list.
        text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        return len(tok.encode(text))

    total_done = 0
    while True:
        ready = scan_complete_ids(in_path, processed)
        if args.limit:
            ready = ready[: max(0, args.limit - total_done)]
        if not ready:
            if args.watch and not args.limit:
                print(f"[idle] no new complete ids, sleeping {args.watch_sleep}s",
                      flush=True)
                time.sleep(args.watch_sleep)
                continue
            break

        for batch_start in range(0, len(ready), args.ids_per_batch):
            batch = ready[batch_start: batch_start + args.ids_per_batch]

            # ---- B1: 16 score prompts per id ----
            b1_messages = []
            b1_meta = []  # (id_, sample_idx)
            id_images = {}  # cache one PIL per id
            for id_, samples in batch:
                rec0 = samples[0]
                gold = rec0["gt_answer"]
                question = rec0["prompt"]
                image_path = rec0["image_path"]
                img = Image.open(image_path).convert("RGB")
                id_images[id_] = img
                for sidx, s in enumerate(samples):
                    text = render_b1(b1_template, s, gold, question)
                    b1_messages.append([{
                        "role": "user",
                        "content": [
                            {"type": "image_pil", "image_pil": img},
                            {"type": "text", "text": text},
                        ],
                    }])
                    b1_meta.append((id_, sidx))

            # Pre-flight oversize guard: drop B1 prompts that won't fit.
            oversized_by_id = {id_: [False] * N_SAMPLES for id_, _ in batch}
            kept_messages, kept_meta = [], []
            for msg, (id_, sidx) in zip(b1_messages, b1_meta):
                try:
                    plen = _msg_token_len(msg)
                except Exception as e:
                    print(f"[WARN] tokenize failed id={id_} sidx={sidx}: {e}", flush=True)
                    plen = b1_budget + 1  # treat as oversized
                if plen > b1_budget:
                    oversized_by_id[id_][sidx] = True
                else:
                    kept_messages.append(msg)
                    kept_meta.append((id_, sidx))
            n_dropped = sum(sum(v) for v in oversized_by_id.values())
            if n_dropped:
                print(f"[skip] dropped {n_dropped}/{len(b1_messages)} oversized B1 prompts", flush=True)

            t0 = time.time()
            b1_kwargs = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_think_b1 else {}
            b1_chat_error = None
            b1_outs = []
            if kept_messages:
                try:
                    b1_outs = llm.chat(kept_messages, sampling_params=sampling_b1, use_tqdm=False, **b1_kwargs)
                except Exception as e:
                    b1_chat_error = repr(e)
                    print(f"[ERROR] B1 llm.chat failed for batch_start={batch_start}: {b1_chat_error}", flush=True)
                    b1_outs = []
            b1_dt = time.time() - t0

            scores_by_id = {id_: [None] * N_SAMPLES for id_, _ in batch}
            raw_by_id    = {id_: [None] * N_SAMPLES for id_, _ in batch}
            ok_by_id     = {id_: [False] * N_SAMPLES for id_, _ in batch}
            samples_by_id = {id_: samples for id_, samples in batch}
            golds_by_id   = {id_: samples[0]["gt_answer"] for id_, samples in batch}
            for (id_, sidx), out in zip(kept_meta, b1_outs):
                text = out.outputs[0].text
                raw_by_id[id_][sidx] = text
                d = parse_b1(text)
                if d is not None:
                    d["sample_idx"] = sidx
                    d["answer_correctness"] = compute_ac(
                        samples_by_id[id_][sidx].get("answer", ""),
                        golds_by_id[id_],
                    )
                    d["score_total"] = score_total(d)
                    scores_by_id[id_][sidx] = d
                    ok_by_id[id_][sidx] = True

            # ---- pick winners ----
            winners = {}  # id_ -> (best_idx, best_score_total or None, all_b1_failed)
            for id_, samples in batch:
                cands = [(i, s) for i, s in enumerate(scores_by_id[id_]) if s is not None]
                if not cands:
                    winners[id_] = (0, None, True)  # default to sample 0, flag failure
                else:
                    best_idx, best_s = max(cands, key=lambda x: (x[1]["score_total"], -x[0]))
                    winners[id_] = (best_idx, best_s["score_total"], False)

            # ---- B2: 1 description per id ----
            b2_messages = []
            for id_, samples in batch:
                best_idx, _, _ = winners[id_]
                winner_sample = samples[best_idx]
                gold = samples[0]["gt_answer"]
                question = samples[0]["prompt"]
                text = render_b2(b2_template, winner_sample, gold, question)
                b2_messages.append([{
                    "role": "user",
                    "content": [
                        {"type": "image_pil", "image_pil": id_images[id_]},
                        {"type": "text", "text": text},
                    ],
                }])

            # Pre-flight oversize guard for B2 (one prompt per id).
            b2_kept, b2_kept_idx, b2_oversized = [], [], set()
            for i, msg in enumerate(b2_messages):
                try:
                    plen = _msg_token_len(msg)
                except Exception:
                    plen = b2_budget + 1
                if plen > b2_budget:
                    b2_oversized.add(i)
                else:
                    b2_kept.append(msg)
                    b2_kept_idx.append(i)
            if b2_oversized:
                print(f"[skip] dropped {len(b2_oversized)}/{len(b2_messages)} oversized B2 prompts", flush=True)

            t1 = time.time()
            b2_kwargs = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_think_b2 else {}
            b2_chat_error = None
            b2_outs_kept = []
            if b2_kept:
                try:
                    b2_outs_kept = llm.chat(b2_kept, sampling_params=sampling_b2, use_tqdm=False, **b2_kwargs)
                except Exception as e:
                    b2_chat_error = repr(e)
                    print(f"[ERROR] B2 llm.chat failed for batch_start={batch_start}: {b2_chat_error}", flush=True)
                    b2_outs_kept = []
            b2_dt = time.time() - t1
            # Re-expand back to per-id list: outs[i] for kept ids, None for oversized
            b2_outs = [None] * len(b2_messages)
            for kept_pos, orig_i in enumerate(b2_kept_idx):
                if kept_pos < len(b2_outs_kept):
                    b2_outs[orig_i] = b2_outs_kept[kept_pos]

            # ---- emit ----
            ok_b1_total = 0
            ok_b2_total = 0
            for (id_, samples), b2_out in zip(batch, b2_outs):
                if b2_out is None:
                    raw_b2 = ""
                    description = ""
                else:
                    raw_b2 = b2_out.outputs[0].text
                    description = strip_native_think(raw_b2)
                best_idx, best_total, all_failed = winners[id_]
                rec = {
                    "id": id_,
                    "best_idx": best_idx,
                    "best_score_total": best_total,
                    "all_b1_failed": all_failed,
                    "scene_description": description,
                    "scores": scores_by_id[id_],
                    "parse_ok_b1": ok_by_id[id_],
                    "parse_ok_b2": bool(description),
                    "oversized_b1": oversized_by_id[id_],
                    "oversized_b2": id_ in {batch[i][0] for i in b2_oversized},
                    "b1_chat_error": b1_chat_error,
                    "b2_chat_error": b2_chat_error,
                    "raw_b1": raw_by_id[id_],
                    "raw_b2": raw_b2,
                }
                out_f.write(json.dumps(rec) + "\n")
                processed.add(id_)
                ok_b1_total += sum(ok_by_id[id_])
                ok_b2_total += int(bool(description))
            out_f.flush()

            total_done += len(batch)
            print(f"[batch] ids={len(batch)} "
                  f"b1_ok={ok_b1_total}/{len(batch)*N_SAMPLES} "
                  f"b2_ok={ok_b2_total}/{len(batch)} "
                  f"b1_dt={b1_dt:.1f}s b2_dt={b2_dt:.1f}s total={total_done}",
                  flush=True)

            if args.limit and total_done >= args.limit:
                break

        if args.limit and total_done >= args.limit:
            break

    print(f"[done] enriched {total_done} ids", flush=True)


if __name__ == "__main__":
    main()
