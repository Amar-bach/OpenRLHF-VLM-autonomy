"""Snapshot audit for Phase C (C2) progress.

Computes aggregate stats over Phase C output, plus pushes example
traces + length histograms to W&B at each milestone so trace quality
can be eyeballed without leaving the UI.
"""
import argparse, json, random, re
from pathlib import Path

WORD_RE = re.compile(r"\w+")
META_LEAK_RE = re.compile(
    r"\b(the (image|picture|photo)|in the image|as (shown|noted|stated)|"
    r"prior|previous|given (trace|reasoning))\b", re.IGNORECASE)
INSUFF_RE = re.compile(r"insufficient", re.IGNORECASE)


def wc(s: str) -> int:
    return len(WORD_RE.findall(s or ""))


def truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + f" …[+{len(s)-n} chars]"


def stats_str(xs):
    if not xs:
        return "n=0"
    xs = sorted(xs)
    mid = xs[len(xs) // 2]
    return f"n={len(xs)}  mean={sum(xs)/len(xs):.0f}  median={mid}  min={xs[0]}  max={xs[-1]}"


def sample_stratified(by_cat, take):
    """by_cat: dict[category] -> list of ids. take: dict[category] -> N.
    Returns flat list of (category, id) in order of categories given."""
    rng = random.Random(0xC2)
    out = []
    for cat, n in take.items():
        ids = by_cat.get(cat, [])
        if not ids:
            continue
        chosen = rng.sample(ids, min(n, len(ids)))
        for i in chosen:
            out.append((cat, i))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-c-jsonl", required=True)
    ap.add_argument("--stage-b-jsonl", required=True)
    ap.add_argument("--stage-a-jsonl", required=True)
    ap.add_argument("--total-expected", type=int, default=40645)
    ap.add_argument("--n-examples", type=int, default=20,
                    help="Total example rows in the wandb table")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-run-id", default=None)
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--wandb-step", type=int, default=None)
    args = ap.parse_args()

    pc_path = Path(args.phase_c_jsonl)
    if not pc_path.exists() or pc_path.stat().st_size == 0:
        print("[audit] Phase C output file empty / missing — nothing to audit")
        return

    # ===== Pass 1: Phase C records =====
    pc_by_id = {}
    cat_landed, cat_missed, cat_insuff, cat_metaleak = [], [], [], []
    think_lens, answer_lens = [], []
    n = parse_ok = lands = insuff = meta_leak = oversize = chat_err = 0
    with pc_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            n += 1
            id_ = r["id"]
            pc_by_id[id_] = r
            t = r.get("c2_think") or ""
            a = r.get("c2_answer") or ""
            ok = bool(r.get("parse_ok_c2"))
            land = bool(r.get("c2_lands_on_gold"))
            ins  = bool(INSUFF_RE.search(a))
            leak = bool(META_LEAK_RE.search(t))
            if ok:                    parse_ok += 1
            if land:                  lands += 1
            if r.get("oversized_c2"): oversize += 1
            if r.get("c2_chat_error"): chat_err += 1
            if t: think_lens.append(wc(t))
            if a: answer_lens.append(wc(a))
            if ins:  insuff += 1
            if leak: meta_leak += 1
            if land:        cat_landed.append(id_)
            elif ins:       cat_insuff.append(id_)
            elif leak:      cat_metaleak.append(id_)
            else:           cat_missed.append(id_)

    # ===== Pass 2: Stage B scene_description + best_idx for ids in pc =====
    sb_by_id = {}
    with Path(args.stage_b_jsonl).open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            id_ = r.get("id")
            if id_ not in pc_by_id:
                continue
            sb_by_id[id_] = {
                "best_idx":          r.get("best_idx", 0),
                "scene_description": r.get("scene_description") or "",
            }

    # ===== Pass 3: Stage A question, gt_answer, winning trace =====
    sa_by_id = {}  # id -> {question, gt_answer, winning_trace}
    needed = set(pc_by_id.keys())
    win_trace_lens = []
    with Path(args.stage_a_jsonl).open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            id_ = r.get("id")
            if id_ not in needed:
                continue
            cur = sa_by_id.setdefault(id_, {"question": "", "gt_answer": "",
                                            "winning_trace": ""})
            if not cur["question"]:
                cur["question"]  = r.get("prompt") or r.get("question") or ""
                cur["gt_answer"] = r.get("gt_answer") or r.get("gold_answer") or ""
            bidx = sb_by_id.get(id_, {}).get("best_idx", 0)
            if r.get("sample_idx") == bidx:
                cur["winning_trace"] = r.get("thinking") or ""
                win_trace_lens.append(wc(cur["winning_trace"]))

    # ===== Aggregate metrics =====
    pct = lambda a, b: (100.0 * a / b) if b else 0.0
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    progress = pct(n, args.total_expected)

    metrics = {
        "n_records":               n,
        "progress_pct":            progress,
        "parse_ok_rate":           pct(parse_ok, n),
        "lands_on_gold_rate":      pct(lands, n),
        "lands_of_parseok":        pct(lands, parse_ok),
        "insufficient_rate":       pct(insuff, n),
        "meta_leak_rate":          pct(meta_leak, n),
        "oversize_count":          oversize,
        "chat_err_count":          chat_err,
        "c2_think_mean_words":     mean(think_lens),
        "c2_answer_mean_words":    mean(answer_lens),
        "win_trace_mean_words":    mean(win_trace_lens),
        "length_ratio_c2_over_win": (mean(think_lens) / mean(win_trace_lens))
                                    if think_lens and win_trace_lens else 0.0,
    }

    print(f"=== Phase C audit @ {n} records ({progress:.1f}% of {args.total_expected}) ===")
    print(f"parse_ok_c2          : {parse_ok}/{n} ({pct(parse_ok, n):.1f}%)")
    print(f"c2_lands_on_gold     : {lands}/{n} ({pct(lands, n):.1f}%)   "
          f"of parse_ok: {pct(lands, parse_ok):.1f}%")
    print(f"INSUFFICIENT answer  : {insuff}/{n} ({pct(insuff, n):.1f}%)")
    print(f"meta-leak in c2_think: {meta_leak}/{n} ({pct(meta_leak, n):.1f}%)")
    print(f"oversized_c2         : {oversize}")
    print(f"c2_chat_error        : {chat_err}")
    print(f"--- trace length (words) ---")
    print(f"c2_think      : {stats_str(think_lens)}")
    print(f"winning trace : {stats_str(win_trace_lens)}")
    print(f"c2_answer     : {stats_str(answer_lens)}")
    print(f"length ratio (c2 / winning) : {metrics['length_ratio_c2_over_win']:.2f}x")

    # ===== W&B push =====
    if not args.wandb_project:
        return

    try:
        import wandb
    except Exception as e:
        print(f"[wandb] import failed: {e}")
        return

    try:
        wandb.init(
            project=args.wandb_project,
            id=args.wandb_run_id,
            name=args.wandb_run_name,
            resume="allow",
            reinit=True,
        )

        # Histograms (cap bins so wandb doesn't choke on long tails).
        log_payload = dict(metrics)
        if think_lens:
            log_payload["hist_c2_think_words"]  = wandb.Histogram(think_lens, num_bins=40)
        if win_trace_lens:
            log_payload["hist_win_trace_words"] = wandb.Histogram(win_trace_lens, num_bins=40)
        if answer_lens:
            log_payload["hist_c2_answer_words"] = wandb.Histogram(answer_lens, num_bins=40)

        # Stratified example table: try to fill 4 buckets equally, then top up
        # with extra landed/missed if a bucket is empty.
        per = max(1, args.n_examples // 4)
        plan = {"landed": per, "missed": per, "insufficient": per, "meta_leak": per}
        picks = sample_stratified(
            {"landed": cat_landed, "missed": cat_missed,
             "insufficient": cat_insuff, "meta_leak": cat_metaleak},
            plan,
        )
        # Top up if we're short of n-examples
        shortfall = args.n_examples - len(picks)
        if shortfall > 0:
            extra_pool = [(c, i) for c, lst in
                          (("landed", cat_landed), ("missed", cat_missed))
                          for i in lst if (c, i) not in picks]
            random.Random(0xC2 + 1).shuffle(extra_pool)
            picks.extend(extra_pool[:shortfall])

        cols = ["step", "category", "id", "lands_on_gold", "parse_ok",
                "c2_answer", "gt_answer", "question",
                "scene_description", "winning_trace", "c2_think",
                "c2_think_words", "win_trace_words", "meta_leak"]
        table = wandb.Table(columns=cols)
        for cat, id_ in picks:
            pc = pc_by_id[id_]
            sb = sb_by_id.get(id_, {})
            sa = sa_by_id.get(id_, {})
            ct = pc.get("c2_think") or ""
            wt = sa.get("winning_trace") or ""
            row = [
                args.wandb_step,
                cat,
                id_,
                bool(pc.get("c2_lands_on_gold")),
                bool(pc.get("parse_ok_c2")),
                truncate(pc.get("c2_answer") or "", 400),
                truncate(sa.get("gt_answer") or "", 400),
                truncate(sa.get("question") or "", 800),
                truncate(sb.get("scene_description") or "", 1500),
                truncate(wt, 2000),
                truncate(ct, 2000),
                wc(ct),
                wc(wt),
                bool(META_LEAK_RE.search(ct)),
            ]
            table.add_data(*row)
        log_payload["examples"] = table

        wandb.log(log_payload, step=args.wandb_step)
        wandb.finish()
        print(f"[wandb] logged step={args.wandb_step} "
              f"(metrics + 3 histograms + {len(picks)} example rows)")
    except Exception as e:
        print(f"[wandb] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
