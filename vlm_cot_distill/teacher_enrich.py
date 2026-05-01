"""
Stage B: DeepSeek-V4-Flash teacher enrichment.

Consumes Stage A's JSONL (one line per (id, sample_idx) from Qwen3-VL-235B
N=16 sampling), groups by id, calls DeepSeek-V4-Flash once per id, writes
one enriched record per id to output JSONL.

Race-safe consumer of an actively-growing input file:
- append-only reads, parse-error on trailing line => wait
- id is "complete" iff 16 samples present AND a strictly-later id seen
- output file IS the source of truth for resume (no sidecar)
- single-instance enforced via fcntl.flock on output file
"""
import argparse, fcntl, json, random, re, sys, time
from pathlib import Path
from collections import defaultdict

N_SAMPLES = 16
THINK_RE  = re.compile(r"<think>(.*?)</think>",   re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def load_processed_ids(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    ids = set()
    with out_path.open() as f:
        for line in f:
            try:
                ids.add(json.loads(line)["id"])
            except Exception:
                pass  # tolerate any partial trailing line from a prior crash
    return ids


def scan_complete_ids(input_path: Path, processed: set):
    """Return list of (id, [16 sample dicts]) for ids that are complete,
    not yet processed, and have a strictly-later id seen in the file."""
    by_id = defaultdict(dict)
    order = []
    with input_path.open() as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.endswith("\n"):
                # in-flight tail line — stop here, don't consume
                break
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                # malformed mid-file line: real bug, surface it
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


def render_prompt(template: str, sample_set, rng: random.Random):
    rec0 = sample_set[0]
    question = rec0["prompt"]
    gold = rec0["gt_answer"]
    candidate_order = list(range(N_SAMPLES))
    rng.shuffle(candidate_order)
    blocks = []
    for orig_idx in candidate_order:
        s = sample_set[orig_idx]
        grounding = (s.get("grounding") or "").strip()
        cot = (s.get("thinking") or s.get("raw") or "").strip()
        ans = (s.get("answer") or "").strip()
        blocks.append(
            f"[orig_idx={orig_idx}]\n"
            f"  Grounding: {grounding}\n"
            f"  Reasoning: {cot}\n"
            f"  Answer:    {ans}"
        )
    rendered = (template
                .replace("{question}", str(question))
                .replace("{gold_answer}", str(gold))
                .replace("{candidates_block}", "\n\n".join(blocks)))
    return rendered, candidate_order


def parse_teacher_output(text: str):
    a = ANSWER_RE.search(text)
    if not a:
        return None
    answer = a.group(1).strip()
    t = THINK_RE.search(text)
    return {
        "think":    t.group(1).strip() if t else "",
        "answer":   answer,
        "rejected": answer.upper() == "REJECT",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, process at most N ids and exit (test mode)")
    ap.add_argument("--watch", action="store_true",
                    help="Loop forever, sleeping when no new complete ids")
    ap.add_argument("--watch-sleep", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-input-tokens", type=int, default=48000)
    ap.add_argument("--max-output-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    template = Path(args.template).read_text()
    rng = random.Random(args.seed)

    out_f = out_path.open("a")
    try:
        fcntl.flock(out_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"another teacher_enrich already holds {out_path}")

    processed = load_processed_ids(out_path)
    print(f"[boot] {len(processed)} ids already enriched in {out_path}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        kv_cache_dtype="fp8",
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        max_model_len=args.max_input_tokens + args.max_output_tokens,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_output_tokens,
    )

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
            else:
                break

        for batch_start in range(0, len(ready), args.batch_size):
            batch = ready[batch_start: batch_start + args.batch_size]
            prompts, orders, ids = [], [], []
            for id_, samples in batch:
                p, candidate_order = render_prompt(template, samples, rng)
                prompts.append([{"role": "user", "content": p}])
                orders.append(candidate_order)
                ids.append(id_)

            t0 = time.time()
            outs = llm.chat(prompts, sampling_params=sampling, use_tqdm=False)
            dt = time.time() - t0

            ok_count = 0
            for id_, candidate_order, out in zip(ids, orders, outs):
                text = out.outputs[0].text
                parsed = parse_teacher_output(text)
                if parsed is not None:
                    ok_count += 1
                rec = {
                    "id": id_,
                    "candidate_order": candidate_order,
                    "raw": text,
                    "parsed": parsed,
                    "parse_ok": parsed is not None,
                }
                out_f.write(json.dumps(rec) + "\n")
                processed.add(id_)
            out_f.flush()
            total_done += len(batch)
            print(f"[batch] n={len(batch)} parse_ok={ok_count}/{len(batch)} "
                  f"dt={dt:.1f}s ({dt/len(batch):.2f}s/ex) total={total_done}",
                  flush=True)

            if args.limit and total_done >= args.limit:
                break

        if args.limit and total_done >= args.limit:
            break

    print(f"[done] enriched {total_done} ids", flush=True)


if __name__ == "__main__":
    main()
