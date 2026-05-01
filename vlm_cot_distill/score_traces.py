"""
Pass 2: score CoT traces produced by `batch_infer.py` using a VL judge model.

Input JSONL: records from `batch_infer.py` (must have id, image_path, prompt,
gt_answer, thinking, answer).

Output JSONL: original fields + `score` dict with the judge's JSON.
"""

import argparse
import json
import re
from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams

SCORE_RE = re.compile(r"<score>\s*(\{.*?\})\s*</score>", re.DOTALL)


def build_judge_messages(
    system_prompt: str,
    image_path: str,
    question: str,
    gt_answer: str,
    student_thinking: str,
    student_answer: str,
) -> list[dict]:
    # Strip the boilerplate format block from the question for readability
    question_clean = re.sub(r"Reason carefully.*", "", question, flags=re.DOTALL).strip()
    user_text = (
        "# Question (the student was asked this)\n"
        f"{question_clean}\n\n"
        "# Ground-truth answer\n"
        f"{gt_answer}\n\n"
        "# Student's `<think>` trace\n"
        f"{student_thinking}\n\n"
        "# Student's `<answer>`\n"
        f"{student_answer}\n\n"
        "Score this trace per the rubric. Return only the <score>...</score> block."
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def parse_score_block(text: str) -> dict | None:
    m = SCORE_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Judge model (local path or HF id)")
    ap.add_argument("--input", required=True, help="Generation-pass JSONL")
    ap.add_argument("--output", required=True, help="Scored JSONL")
    ap.add_argument("--judge-prompt", required=True, help="Path to judge system prompt")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="Judge output budget — scoring doesn't need long thinking")
    ap.add_argument("--temperature", type=float, default=0.1,
                    help="Low temperature for deterministic grading")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--quantization", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()

    system_prompt = Path(args.judge_prompt).read_text()
    rows = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    print(f"Scoring {len(rows)} traces from {args.input}")

    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={"image": 1},
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
        messages = build_judge_messages(
            system_prompt,
            r["image_path"],
            r["prompt"],
            str(r.get("gt_answer", "")),
            r.get("thinking", ""),
            r.get("answer", ""),
        )
        vllm_inputs.append({
            "prompt": llm.get_tokenizer().apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ),
            "multi_modal_data": {"image": image},
        })

    outputs = llm.generate(vllm_inputs, sampling)

    n_parsed = n_kept = 0
    with Path(args.output).open("w") as f:
        for r, o in zip(rows, outputs):
            text = o.outputs[0].text
            score = parse_score_block(text)
            if score:
                n_parsed += 1
                if score.get("keep"):
                    n_kept += 1
            rec = dict(r)
            rec["score"] = score
            rec["score_raw"] = text
            rec["score_parsed"] = score is not None
            rec["score_num_output_tokens"] = len(o.outputs[0].token_ids)
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(rows)} records to {args.output}")
    print(f"Parsed JSON: {n_parsed}/{len(rows)}   keep=true: {n_kept}/{len(rows)}")


if __name__ == "__main__":
    main()
