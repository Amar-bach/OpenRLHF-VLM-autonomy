"""Sample ~500 rows per template_type from SURDS validation.

Usage: python sample_surds_val.py [--per-type N] [--seed S] --output PATH
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SRC = "/mnt/data4/shasta/amar.amarjyoti/research_data/processed/surds/validation_qa.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    by_type = defaultdict(list)
    for line in Path(SRC).read_text().splitlines():
        r = json.loads(line)
        by_type[r["template_type"]].append(r)

    rng = random.Random(args.seed)
    picks = []
    for t, rows in sorted(by_type.items()):
        k = min(args.per_type, len(rows))
        picks.extend(rng.sample(rows, k))
        print(f"  {t:10s}: {k}/{len(rows)}")

    with open(args.output, "w") as f:
        for r in picks:
            f.write(json.dumps({
                "id": r["sample_id"],
                "image_path": r["image_path"],
                "prompt": r["prompt"],
                "gt_answer": r["answer"],
                "task_family": r["task_family"],
                "template_type": r["template_type"],
            }) + "\n")
    print(f"wrote {len(picks)} to {args.output}")


if __name__ == "__main__":
    main()
