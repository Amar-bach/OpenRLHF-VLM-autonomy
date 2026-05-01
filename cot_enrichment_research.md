# CoT Enrichment Research

## TL;DR

1. **Modality bridging beats direct conditioning**: Vision-R1 uses a three-stage pipeline (MLLM pseudo-CoT → enriched image description → text-only reasoning) rather than feeding images directly to the teacher. For your text-only teacher (DeepSeek-V4-Flash), condition on dense captions as proven surrogates, not raw images.

2. **Score rubric order and reference answers matter**: Scoring biases (rubric order, reference answer score, score ID labels) affect judge consistency significantly. Use full-marked reference answers, randomize rubric presentation order, and test both Arabic numerals and letter scales.

3. **Answer conditioning doesn't inherently leak**: STaR and Quiet-STaR show that conditioning on the gold answer during trace generation works—the key is not to include the gold answer in the trace itself. Your prompt can say "gold answer is {gold_answer}" in the context; the judge's job is to generate a forward-looking trace that happens to lead there.

4. **Position bias is real and measurable**: LLM judges systematically favor candidates in earlier positions (accuracy shifts >10% on reordering). Randomize candidate order in your JSON output or use pairwise comparisons for filtering.

5. **G-Eval's CoT-for-scoring scales**: Liu et al. (EMNLP 2023) showed that asking the judge to first generate detailed evaluation steps (CoT) before scoring improves correlation with humans from 0.392 (baseline) to 0.514. Bake reasoning-before-scoring into your single prompt.

6. **Progressive constraints help reasoning**: Vision-R1's Progressive Thinking Suppression Training gradually relaxes length constraints during RL; for a single-prompt teacher, bound response length and complexity upfront (strict JSON schema), then be permissive in the critiquing phase.

7. **Multi-dimensional scoring outperforms single score**: Prometheus 2 and recent rubric work show that separate scores for faithfulness, validity, completeness, and conciseness (with justification per dimension) correlate ~0.89 with human judgment vs. ~0.39 for single-score judges.

8. **Decoupling reasoning from formatting is critical**: Recent work (Deco-G, 2024) shows that strict format constraints degrade reasoning quality. Use JSON mode sparingly—structure the output post-reasoning, not inline.

---

## Vision-R1: Modality Bridging & Filter Pipeline

**Key Findings:**
Vision-R1 (Huang et al., March 2025) constructs a 200K multimodal CoT dataset via modality bridging and data filtering. The pipeline operates in three stages:

1. **Pseudo-CoT Generation**: An MLLM (Qwen3-VL) receives (image, question) and generates reasoning with visual descriptions embedded.
2. **Description Enrichment**: The original image + question + pseudo-CoT are fed back to the MLLM with a prompt like: *"Given image X, question Y, and thinking process Z, provide a detailed description containing all necessary details to answer correctly."* This yields richer language representations than direct captioning.
3. **Text-Only Refinement**: Enriched descriptions are passed to DeepSeek-R1 (text-only, reasoning-specialized) to generate human-like complex reasoning with questioning, reflection, and inspection.

After generation, **rule-based filtering** removes logically inconsistent samples and improves semantic coherence.

Vision-R1 also introduces **Progressive Thinking Suppression Training (PTST)**, which initially suppresses reasoning length during early RL training, then progressively relaxes constraints to allow longer CoT on complex problems. This prevents "overthinking" early on.

**Direct Quotes & Citations:**
- "Modality bridging converts vision information to language...leverages an existing MLLM to generate Pseudo-CoT reasoning text from multimodal image-text pairs" ([Vision-R1 arXiv](https://arxiv.org/abs/2503.06749))
- The PTST strategy "gradually refines the model's ability to learn correct and complex reasoning processes" while initially constraining length ([MarkTechPost summary](https://www.marktechpost.com/2025/03/26/vision-r1-redefining-reinforcement-learning-for-large-vision-language-models/))

**→ Implication for our prompt:**
Your teacher receives dense captions (from Qwen3-VL-235B) as modality proxies. Do NOT ask the teacher to invent image understanding; treat captions as authoritative. Gate your candidates by whether their reasoning is consistent with the caption details.

---

## LLM-as-a-Judge Prompting for Reasoning Traces

**Key Findings:**
Recent surveys (Zhao et al., 2024–2026) and papers on judge LLMs identify best practices:

**Rubric-Based Evaluation (Prometheus Framework):**
Every judge prompt should include:
1. Task instruction (the problem the candidate is solving)
2. Model response (the candidate trace + answer)
3. Reference answer (a gold-standard response structure)
4. Custom scoring rubric (domain-specific criteria)

Prometheus 2 (meta-trained on curated rubrics) achieves Pearson correlation of 0.897 with humans on multi-dimensional scoring, matching GPT-4 (0.882) and far outperforming single-score baselines (ChatGPT: 0.392).

**G-Eval's CoT-for-Scoring (EMNLP 2023):**
Liu et al. show that asking the judge to generate detailed evaluation steps (internal CoT) *before* scoring improves reliability. The prompt structure is:
1. Provide task + rubric.
2. Request: *"Generate a chain-of-thought of detailed evaluation steps based on the criteria above."*
3. Then ask for form-filling scores.

This two-phase approach (reason → score) outperforms direct scoring.

**Multi-Dimensional vs. Single-Score:**
Breaking evaluation into orthogonal dimensions (faithfulness, validity, completeness, conciseness) with per-dimension justification correlates ~0.89 with human judgment. Single-score judges are far noisier.

**Direct Quotes & Citations:**
- "Prometheus scores 0.897 Pearson correlation with human evaluators when evaluating with 45 customized rubrics" ([Prometheus-Eval GitHub](https://github.com/prometheus-eval/prometheus))
- "G-Eval automatically generates detailed evaluation steps using a chain-of-thought mechanism, improving performance" ([G-Eval ACL 2023](https://aclanthology.org/2023.emnlp-main.153/))
- "Instance-Specific Rubrics" with 10–40 bespoke criteria per example outperform static rubrics ([recent rubric work](https://www.montecarlodata.com/blog-llm-as-judge/))

**→ Implication for our prompt:**
Embed CoT-for-scoring: ask the teacher to first critique each candidate (generate reasoning), then assign per-dimension scores (faithfulness, logical validity, completeness vs. conciseness), with a final gate decision. Use JSON mode *after* the reasoning, not inline.

---

## Answer-Conditioned Rationalization Without Leakage

**Key Findings:**
STaR (Zelikman et al., 2022) and follow-ups (Quiet-STaR 2024, V-STaR, rStar-Math 2025) show that conditioning on the gold answer is safe if done correctly:

**STaR's Core Loop:**
1. Generate rationales to answer questions via few-shot CoT: p(r | x)
2. **For incorrect answers**: regenerate rationale given the correct answer: p(r | x, y*) 
3. Fine-tune on all rationales that led to correct answers
4. Repeat with improved model

The key insight is that p(r | x, y*) explores a "better search space" for rationales than p(r | x) alone. The conditioning happens at generation time, not in the trace; the trace reads as forward reasoning.

**Quiet-STaR's Implicit Conditioning:**
Rather than explicit prompting with the answer, Quiet-STaR uses **teacher forcing** during training (ground-truth tokens inserted into the computation graph), but the learned internal rationales are never explicitly prompted at inference. The model learns to condition implicitly.

**V-STaR & rStar-Math:**
These extend STaR by using both correct *and* incorrect reasoning traces (via DPO-style learning) to train verifiers, avoiding the need for explicit conditioning at inference time.

**Direct Quotes & Citations:**
- "Rationalization conditions on the answer, letting the model access p(r | x, y) which may be a better search space for rationales" ([STaR OpenReview](https://openreview.net/pdf?id=_3ELRdg2sgI))
- "V-STaR utilizes both correct and incorrect LLM-generated solutions during iterative self-improvement" ([V-STaR ACL 2024](https://arxiv.org/html/2402.06457v2))
- Quiet-STaR uses "teacher-forcing by assuming the model selected the correct next ground-truth token" ([Quiet-STaR COLM 2024](https://arxiv.org/html/2403.09629v1))

**→ Implication for our prompt:**
You can safely include `{gold_answer}` in the teacher's context (e.g., "gold answer is GSM8K solution: 42"). The teacher's job is to generate a trace that *appears* to arrive at 42 through forward reasoning, not backward inference. Test with and without the answer signal; STaR-based research suggests it helps, not hurts, trace quality.

---

## Single-Prompt Multi-Task Structuring for Reasoning Models

**Key Findings:**
Recent work on structured prompting (2024–2025) identifies a critical challenge: strict format constraints degrade reasoning. 

**Deco-G (Decoupling Framework):**
A 2024 paper shows that separating task reasoning from output formatting improves performance significantly. When task instructions and format instructions are stacked in the prompt, reasoning suffers because the model allocates cognitive resources to format compliance.

**Recommended Approach:**
1. **User prompt**: Task description, context, examples, and all reasoning instructions. Minimal format guidance.
2. **Post-reasoning**: Extract or validate JSON after the model completes reasoning.
3. **JSON Mode as Safety Net**: Use OpenAI-style Structured Output Mode (strict schema validation) *after* generation, not as an inline constraint.

**Multi-Task in Single Prompt:**
Research shows you can structure multi-step tasks (critique + score + gate) in a single prompt if you:
- Use clear delimiters (`---`, section headers) rather than nested JSON.
- Ask for reasoning *first*, then structured output.
- Number steps explicitly (1., 2., 3.) to reduce task confusion.

**Direct Quotes & Citations:**
- "Stricter format constraints generally lead to greater degradation in reasoning performance" ([Deco-G arXiv 2024](https://arxiv.org/html/2510.03595v1))
- "With reinforcement learning plus schema validation, LLMs achieved 30–60% higher accuracy in JSON-conformant outputs" ([Structured Prompting Guide](https://codeconductor.ai/blog/structured-prompting-techniques-xml-json/))

**→ Implication for our prompt:**
Structure as:
```
System: You are a teacher distillation judge...

User:
[Task intro]

[Dense caption]
[Question]
[Gold answer]

[Candidates 1–8 with numbering]

---

Your task:
1. Critique each candidate (free-text reasoning).
2. Derive your own answer-conditioned CoT.
3. Self-check your answer against the gold answer.
4. Score on dimensions [faithfulness, validity, completeness, conciseness].
5. Gate the best trace(s) for SFT.

Output JSON at the end with all scores and decisions.
```
Do not force JSON structure during the reasoning phases.

---

## DeepSeek-R1 / V4-Flash Prompt Engineering Quirks

**Key Findings:**
Official DeepSeek documentation and 2025 blog posts reveal:

**System Prompt Handling:**
DeepSeek-R1 performs best with **minimal or no system prompt**. All instructions should go in the user message. This is opposite to some GPT-4 guidance, where system prompts are idiomatic.

**Thinking Tags:**
R1 models use `<think>` and `</think>` tags. The opening `<think>` is often auto-injected by the interface/template; only the closing `</think>` appears in raw output. Do not ask the model to output thinking tags explicitly—they're automatic.

**Few-Shot Prompting:**
Few-shot examples often *degrade* performance with DeepSeek-R1. The model attempts to mimic your example patterns rather than using its superior reasoning. Use zero-shot or very minimal examples (1–2 max).

**Temperature & Sampling:**
Recommended settings are temperature 0.5–0.7 (0.6 optimal) and top_p 0.95. Avoid temperature >0.8 (incoherent repetition) or <0.5 (shallow reasoning).

**Avoid Step-by-Step Instructions:**
Explicit "think step-by-step" or "reason as follows" instructions reduce effectiveness; the model already does this autonomously.

**Direct Quotes & Citations:**
- "Avoid adding a system prompt; all instructions should be contained within the user prompt" ([Together AI DeepSeek Docs](https://docs.together.ai/docs/prompting-deepseek-r1))
- "Few-shot prompting often degrades performance...the model attempts to mimic the pattern of examples rather than use its superior reasoning" ([Passhulk DeepSeek Guide 2025](https://passhulk.com/blog/deepseek-prompt-engineering-guide-master-r1-v3-models-2025/))
- "Temperature range of 0.5–0.7 prevents endless repetitions or incoherent outputs" ([Helicone Thinking Models Guide](https://www.helicone.ai/blog/prompt-thinking-models))

**→ Implication for our prompt:**
For DeepSeek-V4-Flash (text-only reasoning), use an **empty or minimal system prompt** (e.g., just "You are a teacher evaluator."), and put all task structure in the user message. Avoid examples that might bias reasoning. Let the model's internal reasoning shine.

---

## Known Biases in LLM Judges & Mitigation

**Key Findings:**
Recent systematic studies (2024–2026) document three major scoring biases:

**1. Position Bias:**
LLM judges systematically favor solutions based on their position in the prompt. A 2025 study of 15 judges and 40,000 instances found accuracy shifts exceeding 10% when candidate order is swapped. Quality differences between solutions amplify this bias.

**Mitigation:** Randomize candidate order. Use pairwise comparisons for critical rankings. Introduce metrics like "position consistency" to audit your judge.

**2. Score Rubric & Reference Answer Biases:**
- **Rubric order**: Ascending vs. descending score scales yield different distributions.
- **Reference answer score**: If your reference answer is marked as 5/5, the judge inflates scores; full-marked references work best.
- **Score ID labels**: Using Arabic numerals (1–5) vs. letters (A–E) vs. Roman numerals (I–V) affects scoring patterns.

**Mitigation:** Use full-marked reference answers. Randomize rubric ordering across runs. Test different score label systems; larger models (GPT-4o) are more robust to these biases.

**3. Score Inflation (Agreeableness Bias):**
Judges often assign high scores liberally (TPR >96%) but rarely catch poor outputs (TNR <25%). This creates false confidence in judge agreement.

**Mitigation:** Use ensemble judges or majority voting (if feasible). Use more capable models (GPT-4o > GPT-4 > GPT-3.5). Calibrate score distributions by testing on known bad/good examples.

**Direct Quotes & Citations:**
- "Position bias significantly compromises LLM judge reliability, varying significantly across judges and tasks" ([Judging the Judges IJCNLP 2025](https://aclanthology.org/2025.ijcnlp-long.18.pdf))
- "Score rubric order and reference answer score significantly influence judge behavior" ([Evaluating Scoring Bias 2025](https://arxiv.org/html/2506.22316v1))
- "Position consistency and preference fairness are key evaluation metrics" ([Position Bias Study](https://arxiv.org/abs/2406.07791))

**→ Implication for our prompt:**
1. Randomize the order of 8 candidates in each invocation.
2. Use a full-marked (100%) reference answer in your rubric.
3. Test multiple score scales (1–5, 1–10, Likert) with your judge.
4. Compare outputs across temperature/sampling settings to check consistency.
5. Plan for ensemble judging if filtering is critical (e.g., run the same example with 2–3 different random seeds and aggregate decisions).

---

## CoT Scoring Rubrics: Dimensions & Calibration

**Key Findings:**
Literature on CoT evaluation (FineSurE, G-Eval, Prometheus, TN-Eval) converges on four core dimensions:

| Dimension | Definition | Scale | Notes |
|-----------|-----------|-------|-------|
| **Faithfulness** | Does the reasoning use facts from the context (caption, question)? Are citations/references accurate? | 0–4 | Critical for preventing hallucination |
| **Logical Validity** | Are the steps logically sound? Does conclusion follow from premises? | 0–4 | Detect contradictions, non sequiturs |
| **Completeness** | Does the trace cover all necessary steps to reach the answer? Any hand-waving? | 0–4 | Assess whether reader could verify |
| **Conciseness** | Is the trace free of redundancy/filler? Does length match problem complexity? | 0–4 | Balance against completeness |

**Calibration Strategies:**
- **0–4 scale** is standard for judge LLMs (lower variance than 0–10).
- **Include anchors**: Show 0 (unacceptable), 2 (adequate), 4 (exemplary) examples for each dimension in the rubric.
- **Per-dimension justification**: Require 1–2 sentences of reasoning per score, not just the score itself.
- **Separate "gating" from scoring**: Add a binary decision: "Include this trace in SFT data? Yes/No" after scoring.

**Recent Rubric Work:**
FineSurE (Song et al., 2024) breaks down evaluation into multiple dimensions systematically. The 2024–2026 consensus is that explicit, criterion-separated rubrics with clear anchors outperform implicit rubrics.

**Direct Quotes & Citations:**
- "FineSurE breaks down evaluation into multiple dimensions: faithfulness, completeness, conciseness" ([FineSurE mentioned in judge survey](https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method))
- "Prometheus uses a 0–4 scale rubric with reference examples for each level" ([Prometheus GitHub](https://github.com/prometheus-eval/prometheus))
- "Per-dimension rubrics correlate 0.89 with humans vs. 0.39 for single-score approaches" ([Multi-dimensional evaluation](https://arxiv.org/html/2501.00274v1))

**→ Implication for our prompt:**
Use a 0–4 scale for each of the four dimensions above. Provide anchors (0, 2, 4 examples) in the rubric itself. Ask for per-dimension justification (1–2 sentences). Then ask for a binary gate decision. Output as JSON with structure:
```json
{
  "candidate_1": {
    "faithfulness": {"score": 3, "justification": "..."},
    "validity": {"score": 4, "justification": "..."},
    "completeness": {"score": 2, "justification": "..."},
    "conciseness": {"score": 3, "justification": "..."},
    "include_in_sft": true
  },
  ...
}
```

---

## Recommended Prompt Skeleton

Below is a **starter template** for your single-prompt teacher. Placeholders like `{caption}` should be substituted with actual values. This is a starting point; iterate based on your findings.

```
[SYSTEM MESSAGE - Minimal for DeepSeek-V4-Flash]
You are an expert evaluator of chain-of-thought reasoning in vision-language tasks.

[USER MESSAGE]

---
CONTEXT

Image Description:
{caption}

Question:
{question}

Gold Answer:
{gold_answer}

---
CANDIDATES

Candidate 1:
Reasoning: {candidate_1_cot}
Answer: {candidate_1_answer}

[... repeat for candidates 2–8, presented in randomized order ...]

---
EVALUATION RUBRIC

Score each candidate on four dimensions using a 0–4 scale:

1. **Faithfulness** (0–4): Does the reasoning use only facts from the image description and question? Are all claims grounded?
   - 0: Hallucinations, contradicts given facts
   - 2: Mostly faithful, minor ungrounded claims
   - 4: All reasoning grounded in facts

2. **Logical Validity** (0–4): Are the reasoning steps logically sound? Does the conclusion follow from premises?
   - 0: Contains logical fallacies or contradictions
   - 2: Mostly logical, minor gaps
   - 4: Flawless logical chain

3. **Completeness** (0–4): Are all necessary steps present? Could a reader verify the answer?
   - 0: Major gaps, hand-waving
   - 2: Adequate but skips some justification
   - 4: Thorough, every step explained

4. **Conciseness** (0–4): Is the trace free of redundancy? Does length match problem complexity?
   - 0: Verbose, bloated, repetitive
   - 2: Acceptable, minor redundancy
   - 4: Tight, no waste

---
YOUR TASK

1. **Critique Each Candidate** (free-text reasoning):
   Read candidates 1–8 and for each, write 2–3 sentences identifying strengths and weaknesses.

2. **Derive Your Own Answer-Conditioned CoT** (free-text):
   Given the image description, question, and gold answer, generate a clean, forward-looking chain of thought that:
   - Reads as natural reasoning (not reverse-engineering)
   - Justifies each step from the problem statement
   - Arrives at the gold answer
   - Is concise and clear

3. **Self-Check**:
   Does your CoT answer match the gold answer? If not, revise.

4. **Score Each Candidate**:
   For each candidate, provide:
   - Faithfulness score (0–4) and 1-sentence justification
   - Logical Validity score (0–4) and 1-sentence justification
   - Completeness score (0–4) and 1-sentence justification
   - Conciseness score (0–4) and 1-sentence justification
   - Overall Quality (0–4): average of the four dimensions, rounded
   - Gate Decision: "Include in SFT data?" (Yes/No), with brief reason

5. **Final Output**:
   Return a JSON object with keys for each candidate, structured as shown below.

---
OUTPUT JSON

{
  "teacher_cot": "<your derived CoT from step 2>",
  "teacher_answer": "<final answer>",
  "candidates": {
    "candidate_1": {
      "faithfulness": {"score": 3, "justification": "..."},
      "validity": {"score": 4, "justification": "..."},
      "completeness": {"score": 2, "justification": "..."},
      "conciseness": {"score": 3, "justification": "..."},
      "overall_quality": 3,
      "include_in_sft": true,
      "reason": "..."
    },
    "candidate_2": {...},
    ...
    "candidate_8": {...}
  },
  "summary": {
    "best_candidate": 1,
    "num_sft_ready": 3,
    "most_common_failure_mode": "..."
  }
}
```

---

## Open Questions

1. **How to prevent evaluation-set overfitting?** If you use the same teacher prompt across all examples, will the judge's scoring distribution drift? Should you vary the preamble or system message across runs?

2. **Is answer conditioning actually necessary?** STaR shows it helps, but does DeepSeek-V4-Flash (which is reasoning-specialized) benefit? Have you considered running A/B tests (with/without gold answer in context) on a calibration set?

3. **Multi-candidate ordering effects**: Do later candidates (7–8) get systematically lower scores due to fatigue? Should you split large batches?

4. **Confidence calibration**: The judge's "include in SFT" gate is binary, but would a confidence score (e.g., "include if teacher confidence > 0.7") improve downstream SFT quality? Is there a way to extract confidence from the judge's reasoning?

5. **Trace reconstruction from critiques**: The teacher generates critiques of the 8 candidates, then a new CoT. Should the new CoT be influenced by the critiques (e.g., "avoid mistake found in candidate 3"), or generated independently? Does the former improve quality?

6. **Judge consistency across temperatures**: Preliminary results from position bias studies suggest that temperature matters. Should you run the teacher at a fixed temperature (e.g., 0.6), or ensemble across 2–3 temperatures?

---

## Sources

- [Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models (arXiv 2503.06749)](https://arxiv.org/abs/2503.06749)
- [Vision-R1: Incentivizing Reasoning on Papers With Code](https://paperswithcode.com/paper/vision-r1-incentivizing-reasoning-capability)
- [STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning (OpenReview)](https://openreview.net/pdf?id=_3ELRdg2sgI)
- [rStar-Math: Small LLMs Can Master Math Reasoning (arXiv 2501.04519)](https://arxiv.org/pdf/2501.04519)
- [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking (COLM 2024)](https://arxiv.org/html/2403.09629v1)
- [V-STaR: Training Verifiers for Self-Taught Reasoners (ACL 2024)](https://arxiv.org/html/2402.06457v2)
- [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.153/)
- [Prometheus: Inducing Fine-Grained Evaluation Capability in Language Models (ICLR 2024)](https://arxiv.org/abs/2310.08491)
- [Prometheus GitHub Repo](https://github.com/prometheus-eval/prometheus)
- [A Systematic Study of Position Bias in LLM-as-a-Judge (IJCNLP 2025)](https://aclanthology.org/2025.ijcnlp-long.18.pdf)
- [Judging the Judges: Position Bias Study (arXiv 2406.07791)](https://arxiv.org/abs/2406.07791)
- [Evaluating Scoring Bias in LLM-as-a-Judge (arXiv 2506.22316)](https://arxiv.org/html/2506.22316v1)
- [Decoupling Task-Solving and Output Formatting in LLM Generation (arXiv 2510.03595)](https://arxiv.org/html/2510.03595v1)
- [Together AI DeepSeek R1 Prompting Docs](https://docs.together.ai/docs/prompting-deepseek-r1)
- [DeepSeek R1 Prompt Engineering Guide 2025 (Passhulk)](https://passhulk.com/blog/deepseek-prompt-engineering-guide-master-r1-v3-models-2025/)
- [How to Prompt Thinking Models (Helicone AI Blog)](https://www.helicone.ai/blog/prompt-thinking-models)
- [A Survey on LLM-as-a-Judge (arXiv 2411.15594)](https://arxiv.org/html/2411.15594v6)
- [Rubric-Based Evaluations & LLM-as-a-Judge (Medium, Adnan Masood, April 2026)](https://medium.com/@adnanmasood/rubric-based-evals-llm-as-a-judge-methodologies-and-empirical-validation-in-domain-context-71936b989e80)
- [LLM-Rubric: A Multidimensional, Calibrated Approach (arXiv 2501.00274)](https://arxiv.org/html/2501.00274v1)
- [Structured Prompting Techniques: XML & JSON Guide (CodeConductor)](https://codeconductor.ai/blog/structured-prompting-techniques-xml-json/)
- [Vision-R1 MarkTechPost Summary (March 2025)](https://www.marktechpost.com/2025/03/26/vision-r1-redefining-reinforcement-learning-for-large-vision-language-models/)
- [Fast Quiet-STaR: Thinking Without Thought Tokens (EMNLP 2025 Findings)](https://aclanthology.org/2025.findings-emnlp.1020.pdf)

