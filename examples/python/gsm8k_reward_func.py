"""GSM8K reward function for single-turn PPO training.

This reward extracts:
1) Gold final answer from GSM8K label text ("#### <number>")
2) Predicted final answer from model response (boxed answer preferred, fallback to last number)

It returns binary rewards in {0, 1}.
"""

from __future__ import annotations

import re
from typing import List, Optional

import torch

from openrlhf.utils import extract_boxed_answer


_NUM_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?")
_GSM8K_RE = re.compile(r"####\s*([-+]?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?)")


def _normalize_number(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = text.strip().replace(",", "")
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if value.is_integer():
        return str(int(value))
    return str(value)


def _extract_gold_answer(label: str) -> Optional[str]:
    match = _GSM8K_RE.search(str(label))
    if not match:
        return None
    return _normalize_number(match.group(1))


def _extract_pred_answer(response: str) -> Optional[str]:
    boxed = extract_boxed_answer(response)
    boxed_norm = _normalize_number(boxed)
    if boxed_norm is not None:
        return boxed_norm

    matches = _NUM_RE.findall(response)
    if not matches:
        return None
    return _normalize_number(matches[-1])


def reward_func(queries: List[str], prompts: List[str], labels: List[str], **kwargs) -> dict:
    rewards = []
    valid = 0

    for query, prompt, label in zip(queries, prompts, labels):
        if isinstance(prompt, str) and query.startswith(prompt):
            response = query[len(prompt) :]
        else:
            response = query

        gold = _extract_gold_answer(str(label))
        pred = _extract_pred_answer(response)

        correct = 1.0 if (gold is not None and pred is not None and gold == pred) else 0.0
        if gold is not None:
            valid += 1
        rewards.append(correct)

        print(f"[GSM8K Reward] pred={pred} gold={gold} correct={bool(correct)}")

    rewards_tensor = torch.tensor(rewards, dtype=torch.float)
    accuracy = rewards_tensor.mean() if len(rewards_tensor) > 0 else torch.tensor(0.0)

    return {
        "rewards": rewards_tensor,
        "scores": rewards_tensor,
        "extra_logs": {
            "gsm8k_acc": accuracy,
            "gsm8k_gold_parse_rate": valid / max(len(labels), 1),
        },
    }

