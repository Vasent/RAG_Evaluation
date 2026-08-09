# RAG Evaluation Guide

This project uses two evaluation checkpoints: retrieval is evaluated before generation, and the generated answer is evaluated before it is returned. Low scores can trigger one bounded retry. The scores are diagnostics, not proof that an answer is medically correct.

## Benchmark data and page convention

[`data/evaluation_questions.json`](data/evaluation_questions.json) contains 12 manually verified questions: 10 answerable and 2 intentionally unanswerable. Page numbers are 1-based physical PDF pages and match the `page` metadata created by the ingestion pipeline.

Ground truth is at page level because each record identifies `{source, page}`. When a page produces several chunks, collapse retrieved chunks to unique `(source, page)` pairs while preserving the first occurrence before calculating retrieval metrics. This prevents duplicate chunks from inflating scores.

The image-only `(E)-High-Energy-High-Protein-Diet.pdf` produced no extractable text with the project's current PDF ingestion approach, so it is not used as benchmark ground truth. Add OCR before including it in retrieval evaluation.

| ID | Verified topic | Ground-truth page |
| --- | --- | --- |
| `fitness_001` | Daily carbohydrate range and pre-exercise timing | `exercise-nutrition.pdf`, p. 4 |
| `fitness_002` | Daily protein range and protein timing | `exercise-nutrition.pdf`, p. 5 |
| `fitness_003` | Hydration signs and actions | `exercise-nutrition.pdf`, p. 5 |
| `fitness_004` | High-calorie/high-protein meal habits | `Nutrition, High Cal-Pro_Education_sheet.pdf`, p. 1 |
| `fitness_005` | Peanut Butter Banana Smoothie | `Nutrition, High Cal-Pro_Education_sheet.pdf`, p. 3 |
| `fitness_006` | Double Strength Milk | `Nutrition, High Cal-Pro_Education_sheet.pdf`, p. 4 |
| `fitness_007` | Performance meal wheel | `Sports-Nutrition-Fundamentals-To-Improve-Performance-full-resource-v2.8.pdf`, p. 11 |
| `fitness_008` | Carbohydrate ranges by training load | Same PDF, p. 12 |
| `fitness_009` | UK calcium recommendations | Same PDF, p. 33 |
| `fitness_010` | Caffeine dose, timing, and higher-dose warning | Same PDF, p. 41 |
| `fitness_011` | Creatine loading and maintenance protocol | Unanswerable; no supporting page |
| `fitness_012` | ACL reconstruction rehabilitation program | Unanswerable; no supporting page |

The creatine item is a useful hard negative: the sports nutrition PDF mentions creatine as an example of an ergogenic aid on page 38, but supplies no dosing protocol. A system should not turn that mention into a dosage recommendation.

## Retrieval metrics

The evaluation-session PDF gives these target values:

| Metric | Calculation | Target |
| --- | --- | ---: |
| MRR | Mean of `1 / rank of first relevant result` across questions | `> 0.8` |
| MAP | Mean Average Precision across questions | `> 0.7` |
| Precision | Relevant retrieved pages divided by retrieved pages | `> 0.8` |
| Recall | Relevant retrieved pages divided by all ground-truth relevant pages | `> 0.7` |
| F1 | Harmonic mean of precision and recall | `> 0.7` |
| NDCG | Discounted ranking gain divided by ideal discounted gain | `> 0.8` |

Terminology matters. For one question, `1 / first relevant rank` is **Reciprocal Rank (RR)**; only its mean over multiple questions is **MRR**. Similarly, the average of precision-at-relevant-rank values for one question is **Average Precision (AP)**; only its mean over multiple questions is **MAP**. The UI may show per-question RR or AP, while the benchmark report shows MRR or MAP.

For unanswerable records, the relevant set is empty. Recall, AP, and NDCG have a zero denominator and should be reported as `N/A`, not invented as zero or one. Evaluate those records with abstention accuracy and whether the retriever avoided misleading near-matches.

## Answer metrics

Reference-based metrics are suitable for the benchmark:

| Metric | What it measures | Target |
| --- | --- | ---: |
| ROUGE-1 | Unigram overlap | `> 0.5` |
| ROUGE-2 | Bigram overlap | `> 0.3` |
| ROUGE-L | Longest common subsequence | `> 0.4` |
| Fuzzy match | Typo-tolerant full or partial string similarity | `> 70` out of 100 |
| Semantic similarity | Cosine similarity between answer embeddings | No target is defined in the PDF |

LLM-as-a-judge metrics compare the question, retrieved sources, and answer:

| Metric | Calculation | Target |
| --- | --- | ---: |
| Grounding | Percentage of retrieved sources judged useful for supporting the answer | `> 80%` |
| TP percentage | Supported answer claims divided by all answer claims | `> 90%` |
| FP percentage | Unsupported answer claims divided by all answer claims | `< 10%` |
| Relevancy | Direct/helpful rather than evasive when the answer is present | `1` |

The judge should first decide whether the sources contain an answer, split the answer into atomic factual claims, attach supporting source IDs to each supported claim, identify unsupported claims, and assess each source independently. Calculate percentages in application code from the judge's structured labels rather than asking the model to do arithmetic. A refusal is correct when `answerable` is false; it is non-committal only when the sources actually contain the answer.

## Benchmark evaluation versus live evaluation

The benchmark has trusted reference answers and known relevant pages, so it can calculate all retrieval and reference-answer metrics. Aggregate MRR and MAP only over the full benchmark (or a clearly named subset).

An arbitrary live question has neither a trusted reference answer nor an exhaustive set of relevant pages. Therefore live ROUGE, fuzzy matching, recall, and MAP are normally unavailable. The live UI should show grounding, claim support/hallucination, relevancy, and any LLM-judged retrieval scores as **judge estimates**, not ground truth. Do not generate a reference answer merely to score another generated answer.

## Limitations and safe retries

- ROUGE and fuzzy matching reward shared wording and can miss a correct paraphrase or overlook a changed number.
- Embedding similarity captures meaning better but does not by itself establish factual support.
- Grounding percentage can fall when retrieval includes redundant sources even if one source fully supports the answer.
- The thresholds are teaching targets and should be calibrated before production use.
- An LLM judge can be inconsistent. Using the same model to answer and judge introduces correlated self-evaluation bias. Judge calls also add latency and API cost; cache benchmark results and use one structured judge call per stage where practical.
- Medical and nutrition answers still require appropriate professional judgment; evaluation scores do not replace it.

Use at most one retrieval retry and one generation retry. For low recall, increase `top_k`; for low precision, RR/MRR, or NDCG, rewrite the query or rerank results. For low grounding or TP percentage, or high FP percentage, regenerate with a stricter source-only instruction and remove unsupported claims. If the second attempt still fails, return a cautious refusal or the safest grounded subset together with a `needs_review` status. Never create an unbounded evaluator loop.
