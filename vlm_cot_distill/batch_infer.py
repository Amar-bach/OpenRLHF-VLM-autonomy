"""
Batch inference with Qwen3-VL-Thinking via vLLM offline engine.

Input JSONL: one row per sample, each row:
  {"id": "...", "image_path": "/abs/path.jpg", "prompt": "describe the scene"}

Output JSONL: same id, with:
  {"id": ..., "thinking": "...", "content": "...", "finish_reason": ..., "usage": {...}}

The Qwen3-VL-Thinking chat template wraps reasoning in <think>...</think>.
We split on that tag so `thinking` is the raw CoT and `content` is the final answer.
"""

import argparse
import json
import re
from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
STRUCTURED_SECTIONS = ["perception", "grounding", "inference", "verification", "answer"]


def parse_trace_structured(text: str) -> dict:
    """Parse the 5-section driving-reasoning format.

    Returns {perception, grounding, inference, verification, answer, leftover}.
    Sections may appear inside a <think>...</think> envelope (Qwen3-Thinking) or
    standalone. We extract by tag regardless of the surrounding `<think>` wrapper.
    """
    out = {s: "" for s in STRUCTURED_SECTIONS}
    cursor_text = text
    for s in STRUCTURED_SECTIONS:
        m = re.search(rf"<{s}>(.*?)</{s}>", text, flags=re.DOTALL)
        if m:
            out[s] = m.group(1).strip()
            cursor_text = cursor_text.replace(m.group(0), "", 1)
    # Leftover = everything not inside a known tag; strip stray <think>/</think>
    leftover = re.sub(r"</?think>", "", cursor_text).strip()
    out["leftover"] = leftover
    return out


def parse_trace(text: str) -> tuple[str, str, str]:
    """Return (thinking, answer, leftover).

    Qwen3-Thinking's chat template prepends `<think>\\n` to the assistant
    turn as the generation prefix, so the decoded output starts INSIDE the
    thinking block and only contains the closing </think>. We treat
    everything up to the first </think> as thinking, regardless of whether
    the opening tag is present.
    """
    thinking = ""
    answer = ""
    close_idx = text.find("</think>")
    if close_idx >= 0:
        open_idx = text.find("<think>")
        if 0 <= open_idx < close_idx:
            thinking = text[open_idx + len("<think>"):close_idx].strip()
        else:
            thinking = text[:close_idx].strip()
        remainder = text[close_idx + len("</think>"):]
    else:
        tm = THINK_RE.search(text)
        if tm:
            thinking = tm.group(1).strip()
            remainder = THINK_RE.sub("", text)
        else:
            remainder = text
    am = ANSWER_RE.search(remainder)
    if am:
        answer = am.group(1).strip()
    leftover = ANSWER_RE.sub("", remainder).strip()
    return thinking, answer, leftover


def build_messages(prompt: str, image_path: str, system: str | None) -> list[dict]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ],
    })
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Local path or HF id")
    ap.add_argument("--input", required=True, help="Input JSONL")
    ap.add_argument("--output", required=True, help="Output JSONL")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=16384,
                    help="Output budget; thinking can be long")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--limit-mm-per-prompt-image", type=int, default=1)
    ap.add_argument("--system-prompt", type=str, default=None,
                    help="Optional system prompt; path to .txt or literal string")
    ap.add_argument("--quantization", type=str, default=None,
                    help="vLLM quantization method (e.g. fp8, awq, gptq, bitsandbytes)")
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="Skip torch.compile (fixes pathological compile times on large MoEs)")
    ap.add_argument("--parse-mode", choices=["think_answer", "structured"],
                    default="think_answer",
                    help="think_answer: <think>/<answer> tags. structured: 5-section driving format")
    args = ap.parse_args()

    system_prompt = None
    if args.system_prompt:
        p = Path(args.system_prompt)
        system_prompt = p.read_text() if p.is_file() else args.system_prompt

    rows = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows)} samples from {args.input}")

    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={"image": args.limit_mm_per_prompt_image},
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    llm = LLM(**llm_kwargs)

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    vllm_inputs = []
    for r in rows:
        image = Image.open(r["image_path"]).convert("RGB")
        messages = build_messages(r["prompt"], r["image_path"], system_prompt)
        vllm_inputs.append({
            "prompt": llm.get_tokenizer().apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ),
            "multi_modal_data": {"image": image},
        })

    outputs = llm.generate(vllm_inputs, sampling)

    with Path(args.output).open("w") as f:
        for r, o in zip(rows, outputs):
            text = o.outputs[0].text
            rec = {
                "id": r.get("id"),
                "image_path": r["image_path"],
                "prompt": r["prompt"],
                "gt_answer": r.get("gt_answer"),
                "task_family": r.get("task_family"),
                "template_type": r.get("template_type"),
                "raw": text,
                "finish_reason": o.outputs[0].finish_reason,
                "num_prompt_tokens": len(o.prompt_token_ids),
                "num_output_tokens": len(o.outputs[0].token_ids),
                "parse_mode": args.parse_mode,
            }
            if args.parse_mode == "structured":
                rec.update(parse_trace_structured(text))
            else:
                thinking, answer, leftover = parse_trace(text)
                rec["thinking"] = thinking
                rec["answer"] = answer
                rec["leftover"] = leftover
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(rows)} records to {args.output}")


if __name__ == "__main__":
    main()
