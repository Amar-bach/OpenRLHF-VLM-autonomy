"""N-sampling CoT generation for distillation.

Generates N traces per input prompt using vLLM's batched n-sampling.
Output: one JSONL line per (id, sample_idx). Resume-safe.
"""
import argparse
import json
import re
from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
GROUNDING_RE = re.compile(r"<grounding>(.*?)</grounding>", re.S)


def parse_trace(text: str):
    grounding_m = GROUNDING_RE.search(text)
    grounding = grounding_m.group(1).strip() if grounding_m else ""
    rest = GROUNDING_RE.sub("", text)

    thinking = ""
    close_idx = rest.find("</think>")
    if close_idx >= 0:
        open_idx = rest.find("<think>")
        thinking = (
            rest[open_idx + len("<think>"): close_idx]
            if 0 <= open_idx < close_idx
            else rest[:close_idx]
        ).strip()
        remainder = rest[close_idx + len("</think>"):]
    else:
        m = THINK_RE.search(rest)
        if m:
            thinking = m.group(1).strip()
            remainder = THINK_RE.sub("", rest)
        else:
            remainder = rest
    am = ANSWER_RE.search(remainder)
    answer = am.group(1).strip() if am else ""
    leftover = ANSWER_RE.sub("", remainder).strip()
    return grounding, thinking, answer, leftover


def load_done(path: Path):
    done = set()
    if path.exists():
        for line in path.open():
            try:
                r = json.loads(line)
                done.add((r["id"], r["sample_idx"]))
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--system-prompt", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--tensor-parallel-size", type=int, default=8)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--flush-every", type=int, default=50)
    args = ap.parse_args()

    sys_prompt = Path(args.system_prompt).read_text().strip()
    inputs = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    if args.limit:
        inputs = inputs[: args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    if done:
        print(f"[resume] {len(done)} (id,sample_idx) pairs already in {out_path}", flush=True)

    pending = [r for r in inputs if any((r["id"], k) not in done for k in range(args.n))]
    print(f"[run] {len(pending)} / {len(inputs)} prompts have unfinished samples", flush=True)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        quantization=args.quantization,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )

    sampling = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    out_f = out_path.open("a")
    written = 0
    for batch_start in range(0, len(pending), args.flush_every):
        batch = pending[batch_start: batch_start + args.flush_every]
        chat_inputs = []
        for r in batch:
            img = Image.open(r["image_path"]).convert("RGB")
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [
                    {"type": "image_pil", "image_pil": img},
                    {"type": "text", "text": r["prompt"]},
                ]},
            ]
            chat_inputs.append(messages)

        outputs = llm.chat(chat_inputs, sampling_params=sampling, use_tqdm=True)
        for r, out in zip(batch, outputs):
            for k, comp in enumerate(out.outputs):
                if (r["id"], k) in done:
                    continue
                raw = comp.text
                grounding, thinking, answer, leftover = parse_trace(raw)
                rec = {
                    "id": r["id"],
                    "sample_idx": k,
                    "image_path": r["image_path"],
                    "prompt": r["prompt"],
                    "gt_answer": r["gt_answer"],
                    "task_family": r.get("task_family"),
                    "template_type": r.get("template_type"),
                    "raw": raw,
                    "grounding": grounding,
                    "thinking": thinking,
                    "answer": answer,
                    "leftover": leftover,
                    "finish_reason": comp.finish_reason,
                    "num_output_tokens": len(comp.token_ids),
                }
                out_f.write(json.dumps(rec) + "\n")
                written += 1
        out_f.flush()
        print(
            f"[progress] {batch_start + len(batch)}/{len(pending)} prompts, "
            f"{written} new samples written",
            flush=True,
        )

    out_f.close()
    print(f"[done] wrote {written} samples to {out_path}", flush=True)


if __name__ == "__main__":
    main()
