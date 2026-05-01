# CoT Enrichment Plan (Vision-R1-style, single-prompt)

## Goal
Convert N=16 sampled rationales from Qwen3-VL-235B-Thinking-FP8 into one
high-quality answer-conditioned CoT per example, suitable for cold-start
SFT of a smaller VLM (downstream: SURDS / CODA-LM / DriveLM). Collapse
triage, critique, re-derivation, self-check, scoring, and gating into a
SINGLE teacher call (DeepSeek-V4-Flash, text-only).

Research basis: see `cot_enrichment_research.md`.

## Pipeline

### Stage A — VLM sampling (in-flight)
Job 1058163 on amxnl031 (sxm5, 4×H200, started 2026-04-30).
Qwen3-VL-235B-A22B-Thinking-FP8 produces N=16 (cot, answer) candidates per
(image, question) over 41,080 SURDS prompts. Output:
`/mnt/data4/shasta/amar.amarjyoti/research_data/vlm_cot_distill/cot_1058163_Qwen3-VL-235B-A22B-Thinking-FP8_train_N16_T0.8_grounding.jsonl`
One line per (id, sample_idx). Schema includes `id, sample_idx, prompt,
gt_answer, thinking, answer, raw, ...`.

### Stage A.5 — Modality bridging — SKIPPED (open question for later)
Vision-R1's recipe re-prompts the VLM to produce an authoritative
enriched caption. We skip it for the pilot: with N=16, cross-candidate
consensus on visual claims acts as the modality bridge. Revisit if
Stage D audit shows faithfulness scores are noisy or hallucinations slip
through.

### Stage B — Single-prompt teacher enrichment (DeepSeek-V4-Flash)
Per example: one call, inputs are `question`, `gold_answer`, and the 16
candidate (cot, answer) pairs in randomized order to mitigate position
bias. Output is one JSON object (schema below).

**Concurrency with Stage A** (race-safe consumer of an actively-growing
JSONL):
- Append-only reads; trailing parse-error => wait.
- An id is "complete" iff (a) all 16 samples are present AND (b) a
  strictly-later id has been seen in the file. Stage A's `cot_sample.py`
  writes all 16 samples for one id contiguously then moves on, so this
  rule is bulletproof.
- The output JSONL itself is the resume source of truth (no sidecar);
  on startup, scan it once to build the processed-id set.
- Single-instance enforced via `fcntl.flock` on the output file.
- Exit when Stage A's job is gone (`sacct -j 1058163`) AND no
  complete-unprocessed ids remain.

**Serving**: 4×H200 sxm5, vLLM:
```
--data-parallel-size 4 --enable-expert-parallel --kv-cache-dtype fp8
--tokenizer-mode deepseek_v4
```
Sampling: `temperature=0.6, top_p=0.95`.

### Stage C — Filter & format
Apply gate (below). Format kept records into the student VLM's chat
template (image + question → `<think>...</think>` + answer). Rejected
records go to a separate shard for inspection — don't discard.

### Stage D — Audit (must run before scaling)
On a 100-example probe:
- **Position consistency**: re-run with two random candidate-order
  permutations; flag if `keep_for_sft` flips or scores swing >1 point.
- **Score inflation**: hand-label 50 known-bad and 50 known-good traces;
  measure judge TPR / TNR. Recalibrate gate thresholds if TNR < 0.5.
- **Forward-reasoning leak check**: grep enriched CoTs for "since the
  answer is", "given that the answer", "we know the answer"; eyeball
  flagged samples.

## Output schema (per id, written by Stage B)
The teacher emits a fenced ```json block; we wrap it with run metadata:
```json
{
  "id": "<surds id>",
  "candidate_order": [<orig_idx as shown to teacher>, ...],   // length 16
  "raw": "<full teacher text incl. </think>>",
  "parse_ok": true,
  "parsed": {
    "candidate_analysis": [
      {"orig_idx": <int>, "answer_correct": <bool>, "trace_quality": 0-4, "notes": "<1 sentence>"}
    ],
    "enrichment_strategy": "refine|rewrite|synthesize|reject",
    "selected_candidate_idx": <int or -1>,
    "enriched_cot": "<forward-reasoning trace>",
    "final_answer": "<teacher's answer>",
    "self_check_matches_gold": <bool>,
    "scores": {
      "faithfulness":     {"score": 0-4, "justification": "..."},
      "logical_validity": {"score": 0-4, "justification": "..."},
      "completeness":     {"score": 0-4, "justification": "..."},
      "conciseness":      {"score": 0-4, "justification": "..."}
    },
    "keep_for_sft": <bool>,
    "reject_reason": <string or null>
  }
}
```
`candidate_order[shown_position] = original_sample_idx` — needed to map
the teacher's `orig_idx` references back to Stage A `sample_idx`.

## Gate (initial; recalibrate after Stage D)
Keep iff ALL of:
- `parsed.self_check_matches_gold == true`
- `parsed.enrichment_strategy != "reject"`
- `parsed.scores.faithfulness.score >= 3`
- `parsed.scores.logical_validity.score >= 3`
- `min(parsed.scores.completeness.score, parsed.scores.conciseness.score) >= 2`

## Key design decisions (with rationale)
- **Skip A.5**: N=16 cross-candidate consensus replaces the explicit
  enriched caption. Reconsider if faithfulness scoring is unstable.
- **Empty system prompt + zero-shot**: DeepSeek-R1 family degrades with
  system prompts and few-shot examples.
- **0–4 scale, per-dimension justification**: Prometheus / FineSurE
  convention; correlates ~0.89 with humans vs ~0.39 single-score.
- **Reason-first, JSON-last**: strict inline JSON degrades reasoning
  (Deco-G); we ask for prose for steps 1–4 and a single fenced JSON
  block at step 5.
- **Randomized candidate order per call**: position bias mitigation
  (>10% accuracy swing in published studies).
- **Raw output capture + our own JSON extractor**: no `--reasoning-parser`
  dependency; portable across vLLM versions.

## Deliverables
- `vlm_cot_distill/cot_sample.py` (Stage A — already running)
- `vlm_cot_distill/teacher_enrich.py` (Stage B)
- `vlm_cot_distill/prompts/teacher_enrichment.txt` (single-prompt template)
- `slurm_scripts/pretrain_model_10.sh` (Stage B SLURM, test mode)
- `data/enriched/{kept,rejected}.jsonl` (Stage C)
- `notebooks/audit_enrichment.ipynb` (Stage D)

## Open questions
- A.5 A/B: is the modality-bridge enrichment worth ~2× VLM cost?
- Including `gold_answer` in teacher context vs. only at self-check time
  (STaR-line evidence says including helps).
- Single call vs. seed-ensemble (doubles cost; biggest robustness win).
- Token budgets: Stage A's `max-tokens=16384` per candidate means a
  single id's 16 candidates can exceed 200k tokens worst-case. Current
  `max-input-tokens=48000` will truncate; need empirical input-length
  distribution from Stage A output to tune.
