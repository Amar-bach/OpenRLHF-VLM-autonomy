# CoT Enrichment Plan — Qwen Judge + DeepSeek Text-Only Polish

Three-stage pipeline that converts raw Qwen3-VL-235B N=16 sampling traces
(Stage A) into SFT-ready VLM CoT data through:

1. **Stage B — Qwen-as-judge rejection sampling** picks the best of 16
   candidate (grounding + trace) per QA and produces a self-contained text-only
   scene description.
2. **Phase C — DeepSeek-V4-Flash text-only polish** takes the winning trace
   (or the description alone) and produces a clean modality-bridged trace.
3. **Stage D — SFT formatting** emits image-grounded and text-only training
   pairs.

Downstream targets: SURDS, CODA-LM, DriveLM.

---

## Stage A — Sampling (in-flight, unchanged)

- **Model**: Qwen3-VL-235B-A22B-Thinking-FP8 on 4×H200 sxm5
- **Input**: 41k SURDS train QAs (image + question + gold answer)
- **Output per (id, sample_idx)**: `grounding` (`<obj1>desc [x, y]</obj1>...` point-style), `thinking` (CoT), `answer`
- **Job**: 1058163 (running on `amxnl031`, ~1.5 days remaining as of 2026-05-01)
- **Output**: `cot_1058163_Qwen3-VL-235B-A22B-Thinking-FP8_train_N16_T0.8_grounding.jsonl`

---

## Stage B — Qwen-as-judge rejection sampling

**Judge model**: Qwen3-VL-235B-A22B-Thinking-FP8 (same model as Stage A,
**separate** vLLM instance on a second 4×H200 sxm5 allocation).
**Runs concurrently** with Stage A using the race-safe consumer contract.

**Why Qwen judges itself**:
- Has image access — can verify visual grounding claims (DeepSeek text-only could not).
- Same model that generated has matching capability ceiling.
- Self-consistency bias is anchored by gold answer (`answer_correctness`) and
  image grounding (`hallucination` and `visual_grounding`).

**Two call types per QA**:

### Call B1 — Per-trace scoring (16 calls per QA)

**Inputs**: `image, question, gold_answer, grounding_i, trace_i, answer_i`
(one candidate at a time; the judge does NOT see the other 15 candidates).

**Grounding handling — explicit and load-bearing**: the prompt must:
- present `grounding_i` verbatim as `<obj1>desc [x, y]</obj1>` lines,
- instruct the judge to (a) verify each `<objN>` description against the
  image region near `[x, y]`, and (b) check that the trace's spatial
  claims are consistent with the cited coordinates,
- score `visual_grounding` based on per-object correctness:
  description match AND coordinate plausibility.

**Output schema** (one JSON per call, in a fenced block):
```json
{
  "hallucination":      {"score": 1-5, "note": "<one sentence>"},
  "visual_grounding":   {"score": 1-5, "note": "<one sentence; cite which <objN> is wrong if any>"},
  "reasoning_quality":  {"score": 1-5, "note": "<one sentence>"},
  "answer_correctness": 0 or 1
}
```

**Score scale (1–5, anchored)**:

| Score | hallucination | visual_grounding | reasoning_quality |
|---|---|---|---|
| 1 | heavy fabrication of objects/relations not in image | most `<objN>` wrong (description or coords) | incoherent / fallacy / contradicts itself |
| 2 | one or two clear fabrications | a few `<objN>` wrong | step skipped, weak inference |
| 3 | minor unsupported claim | one `<objN>` mildly off | mostly sound, one weak step |
| 4 | nothing fabricated, all claims grounded | all `<objN>` correct, coords approximate | clean derivation, minor padding |
| 5 | every visual claim verifiable in image | every `<objN>` accurate, coords precise | every step a clean inference, tight |

`answer_correctness`: 1 iff the candidate's final answer matches the gold
answer's meaning (allow paraphrase / capitalization / punctuation
differences); else 0.

### Call B2 — Scene description (1 call per QA, after B1 picks the winner)

**Winner selection**: `best_idx = argmax over scores[i].total` where
`total = hallucination + visual_grounding + reasoning_quality + 5 * answer_correctness`
(the `5 *` makes correctness a hard tie-breaker). On tie: lowest `sample_idx`.

**Inputs**: `image, question, gold_answer, winning_grounding, winning_trace, winning_answer`.

**Output**: 4–8 sentence text-only scene description that:
- enumerates **every grounded object** from `winning_grounding` by description and approximate coordinate,
- describes spatial relationships needed for the question (left/right, depth, occlusion, in-front-of, contained-in, etc.),
- contains all visual evidence required to derive the gold answer **without seeing the image**.

**Self-containment requirement**: a text-only reasoner reading
description + question alone must derive the gold answer. This is the
Vision-R1 modality-bridging artifact, used as input to Phase C.

### Per-QA Stage B output record
```json
{
  "id": "<stage-A id>",
  "best_idx": <int 0..15>,
  "scene_description": "<4-8 sentences, enumerates grounded objects + coords>",
  "scores": [
    {"sample_idx": 0, "hallucination": {...}, "visual_grounding": {...},
     "reasoning_quality": {...}, "answer_correctness": 0|1, "score_total": <int>},
    ... (16 entries)
  ],
  "best_score_total": <int>,
  "raw_b1": [<16 raw judge generations>],
  "raw_b2": "<full scene-description generation>",
  "parse_ok_b1": <bool[16]>,
  "parse_ok_b2": <bool>
}
```

---

## Phase C — DeepSeek-V4-Flash text-only polish

**Current scope: C2 only (description-only rederivation).** Stage B finished
2026-05-12 with 40,645 records; Phase C runs the C2 pass over those records
to produce text-only SFT data. **C1 (winner-seeded polish) is parked as TODO**
— see "TODO — C1" section at the bottom of this stage. Reason: leak risk
(DeepSeek's `<think>` referencing the prior trace instead of refining it)
needs prompt/audit work, and C2 is the harder signal to land first
(modality-bridging self-containment test).

DeepSeek is text-only, so it operates on the scene description rather than
the image. Originally specced as two parallel call variants per QA:

**Key design rule — gold answer is NEVER shown to DeepSeek in either call.**
DeepSeek must derive the answer itself. We then check `derived_answer ≈ gold`
deterministically (`compute_ac` reused from Stage B). This guarantees forward
reasoning by construction (no possible peek) and turns the lands-on-gold
check into a real signal of trace honesty.

DeepSeek-V4-Flash is a thinking model: it natively emits
`<think>...reasoning...</think>` followed by free text. We treat the
content of `<think>...</think>` as the trace and ask only for an
`<answer>` block in the prompt. No `<reasoning>` tag is requested
(would be redundant and the model usually ignores it anyway).

### Call C1 — Image-anchored trace (winner-seeded) — TODO, deferred
See "TODO — C1" at the end of this stage. Originally specced as:
inputs `question, scene_description, winning_trace, winning_grounding`,
task to derive answer using description, trace as hint with verify-don't-copy
framing. Deferred for leak-risk mitigation (DeepSeek may reference the
seed trace meta-style instead of refining it).

### Call C2 — Description-only trace (ACTIVE)
**Inputs**: `question, scene_description`. **No gold answer, no trace.**
**Task**: derive the answer using only the scene description. If the
description is insufficient, the model is instructed to say so rather
than hallucinate — that fails the gate honestly.
**Output**: `<answer>...</answer>`; trace harvested from native `<think>`.
**Used for**: text-only SFT split (Stage D2).

### Outcome matrix per id
| C1 lands on gold | C2 lands on gold | Action |
|---|---|---|
| ✓ | ✓ | Both SFT splits emitted |
| ✓ | ✗ | Image-grounded only (description was insufficient) |
| ✗ | ✓ | Text-only only (winning trace was misleading) |
| ✗ | ✗ | Drop |

### Per-QA Phase C output record (trimmed)
```json
{
  "id": "<stage-A id>",
  "c1_think":         "<content of native <think>...</think> in C1 output>",
  "c1_answer":        "<C1 derived answer>",
  "c1_lands_on_gold": <bool>,
  "c2_think":         "<content of native <think>...</think> in C2 output>",
  "c2_answer":        "<C2 derived answer>",
  "c2_lands_on_gold": <bool>,
  "raw_c1":           "<full DeepSeek output for C1>",
  "raw_c2":           "<full DeepSeek output for C2>"
}
```

Skipped ids (Stage B `parse_ok_b2=False`, `all_b1_failed=True`, or empty
`scene_description`) are not written. Empty `c1_think` + non-empty
`raw_c1` indicates a parse failure on a successful chat call. Empty both
indicates an oversize-skip or vLLM chat error (root cause in SLURM log).

---

## Stage D — SFT formatting

For each QA where the gates pass, emit two SFT records (independent splits):

### D1 — Vision-grounded split (image-conditioned student)
```
user:      <image> + question
assistant: <reasoning> {c1_think} </reasoning> <answer> {c1_answer} </answer>
```
**Gate**: `c1_lands_on_gold AND best_score.answer_correctness == 1
       AND best_score.hallucination >= 4
       AND best_score.visual_grounding >= 4
       AND best_score.reasoning_quality >= 3`

### D2 — Text-only split (modality-bridged student)
```
user:      {scene_description} + question
assistant: <reasoning> {c2_think} </reasoning> <answer> {c2_answer} </answer>
```
**Gate**: `c2_lands_on_gold AND parse_ok_b2 AND best_score.visual_grounding >= 4`
(visual_grounding gate ensures the description was built from accurate grounding.)

---

## Audit — `notebooks/cot_enrichment.ipynb`

Updated for new schema:
- Per-QA viewer: image + question + gold + 16 traces with their scores
  (color-coded by total) + scene_description + C1 polished + C2 rederived.
- Distributions: per-dim score histograms, `answer_correctness=1` rate per
  task family, B2 self-containment rate (= C2 lands-on-gold rate).
- Quality flags: scene descriptions missing grounded objects from the
  winner, C1/C2 traces that reference "the image" (text-only leak), Stage A
  cases where 0/16 candidates score `answer_correctness=1` (likely gold
  mislabel).

---

## Compute & timeline

| Stage | Calls | Hardware | Wall-clock |
|---|---|---|---|
| A (in-flight) | 41k × 16 = 656k | 4×H200 sxm5 (1058163) | ~1.5 days remaining |
| B1 + B2 | 41k × 17 = 697k | second 4×H200 sxm5 | ~3.5 days, concurrent with A |
| C2 only (active) | 40,645 × 1 = 40.6k | DeepSeek-V4-Flash 4×H200 sxm5 | ~0.5–1 day |
| C1 (deferred TODO) | 40,645 × 1 = 40.6k | DeepSeek-V4-Flash 4×H200 sxm5 | future pass |

**Total**: ~5 days from now if Stages A/B run concurrent and C queues after B.

---

## Race-safe consumer contract (used by both B and C)

- Read upstream `.jsonl` append-only.
- Parse-error on trailing line ⇒ wait, do not consume.
- An id is "complete" iff:
  - For Stage B reading Stage A: 16 samples present in file AND a strictly-later id has been observed.
  - For Phase C reading Stage B: the id's record has appeared (Stage B writes one record per id).
- Output `.jsonl` IS the resume source of truth (no sidecar).
- Single-instance enforced via `fcntl.flock` on output file.

---

## File layout

```
vlm_cot_distill/
├── stage_a_sample/                 # Qwen3-VL-235B N=16 candidate sampler
│   ├── cot_sample.py
│   └── prompts/
│       ├── system_surds.txt
│       ├── system_surds_structured.txt
│       └── system_surds_grounding.txt
├── stage_b_judge/                  # Qwen3-VL-32B per-trace judge + scene description
│   ├── qwen_judge.py
│   └── prompts/
│       ├── judge_per_trace.txt     # B1
│       └── scene_description.txt   # B2
├── stage_c_polish/                 # DeepSeek-V4-Flash text polish (queued)
│   ├── teacher_enrich.py           # to be replaced by deepseek_polish.py
│   └── prompts/
│       ├── teacher_enrichment_v0.1.txt
│       └── teacher_enrichment_v2.txt
├── data/                           # static fixtures (example inputs, val splits)
├── scripts/                        # one-off bash (model downloads, etc.)
├── tools/                          # data-prep helpers
└── legacy/                         # superseded scripts kept for reference
```

DeepSeek artifacts to retire after Phase C lands:
- `stage_c_polish/teacher_enrich.py`
- `stage_c_polish/prompts/teacher_enrichment_v*.txt`
- `slurm_scripts/pretrain_model_10.sh`

(Keep `vllm_deepseekv4_cu130.sif` — needed for Phase C.)

New SLURM scripts:
- `slurm_scripts/pretrain_model_11.sh` — Stage B (Qwen judge), 4×H200 sxm5
- `slurm_scripts/pretrain_model_12.sh` — Phase C (DeepSeek polish), 4×H200 sxm5

---

## Locked decisions

| # | Decision | Choice |
|---|---|---|
| 1 | B2 trace seed | **winning trace from B1** |
| 2 | Score scale | **1–5** for hallucination / visual_grounding / reasoning_quality; **0/1** for answer_correctness |
| 3 | Hardware | **second 4×H200 sxm5 NOW**, Stage B concurrent with A |
| 4 | Grounding emphasis | **explicit per-`<objN>` verification** in B1; **enumerate every grounded object with coords** in B2 |
| 5 | DeepSeek role | **Phase C**, two parallel calls per QA: C1 polish (anchored on winner), C2 rederive (description-only) |
| 6 | Self-consistency bias | ship as-is; gold + image anchors are sufficient. Revisit if Stage D quality is poor. |
| 7 | DeepSeek artifact retirement | retire after Phase C smoke test passes. |

---

## Status

- [x] Stage A running (job 1058163, ~1.5 days remaining)
- [ ] B1 prompt drafted (`judge_per_trace.txt`) — explicit `<objN>` verification
- [ ] B2 prompt drafted (`scene_description.txt`) — enumerates grounded objects
- [ ] `qwen_judge.py` implemented (race-safe consumer + B1 batch + B2 per-QA)
- [ ] `pretrain_model_11.sh` SLURM wrapper
- [ ] Stage B smoke test (10 QAs)
- [ ] Stage B production launch (concurrent with A)
- [x] C2 prompt drafted (`stage_c_polish/prompts/deepseek_rederive.txt`)
- [x] `deepseek_polish.py` implemented (C2-only)
- [x] `pretrain_model_12.sh` SLURM wrapper
- [ ] Phase C (C2) production launch — 40,645 ids
- [ ] **TODO — C1**: draft `deepseek_polish.txt`, extend `deepseek_polish.py` with C1 path, add leak-detector flag, audit. Pick up after C2 results are in.
- [ ] `cot_enrichment.ipynb` updated for new schema
- [ ] Stage D — SFT formatter + gates
