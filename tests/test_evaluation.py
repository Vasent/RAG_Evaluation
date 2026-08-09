"""Unit tests for the dependency-free RAG evaluation helpers."""

from __future__ import annotations

import math
import unittest

from app.evaluation import (
    average_precision_at_k,
    binary_relevance,
    cosine_similarity,
    derive_answer_metrics,
    evaluate_combined_judgment,
    evaluate_search_metrics,
    extract_json_object,
    f1_score,
    fuzzy_partial_ratio,
    fuzzy_ratio,
    levenshtein_distance,
    ndcg_at_k,
    normalize_combined_judgment,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    rouge_l,
    rouge_n,
    rouge_scores,
    search_grades_from_judgment,
)


class SearchMetricTests(unittest.TestCase):
    def test_pdf_mrr_and_average_precision_example(self) -> None:
        grades = [0, 3, 0, 3, 3, 0, 0, 3, 0, 0]

        self.assertAlmostEqual(reciprocal_rank(grades), 0.5)
        self.assertAlmostEqual(average_precision_at_k(grades), 0.525)
        self.assertAlmostEqual(precision_at_k(grades), 0.4)

    def test_pdf_ndcg_example(self) -> None:
        grades = [0, 3, 0, 3, 3]
        self.assertAlmostEqual(ndcg_at_k(grades), 0.6797310500037655)

    def test_precision_recall_and_f1_example(self) -> None:
        grades = [3, 2, 3, 0, 0]
        precision = precision_at_k(grades)
        recall = recall_at_k(grades, total_relevant=10)

        self.assertAlmostEqual(precision, 0.6)
        self.assertAlmostEqual(recall, 0.3)
        assert recall is not None
        self.assertAlmostEqual(f1_score(precision, recall), 0.4)

        summary = evaluate_search_metrics(grades, total_relevant=10)
        self.assertAlmostEqual(summary["precision_at_k"], 0.6)
        self.assertAlmostEqual(summary["recall_at_k"], 0.3)
        self.assertAlmostEqual(summary["f1"], 0.4)

    def test_threshold_k_and_graded_ndcg(self) -> None:
        grades = [3, 1, 2, 0]
        self.assertEqual(binary_relevance(grades), [1, 0, 1, 0])
        self.assertEqual(precision_at_k(grades, k=2), 0.5)
        self.assertEqual(reciprocal_rank(grades, k=0), 0.0)
        self.assertGreater(ndcg_at_k(grades, graded=True), 0.0)

    def test_search_empty_input_guards(self) -> None:
        self.assertEqual(precision_at_k([]), 0.0)
        self.assertEqual(reciprocal_rank([]), 0.0)
        self.assertEqual(average_precision_at_k([]), 0.0)
        self.assertEqual(ndcg_at_k([]), 0.0)
        self.assertIsNone(recall_at_k([], total_relevant=0))

        summary = evaluate_search_metrics([])
        self.assertEqual(summary["retrieved_count"], 0)
        self.assertIsNone(summary["recall_at_k"])
        self.assertIsNone(summary["f1"])

    def test_invalid_search_arguments(self) -> None:
        with self.assertRaises(ValueError):
            precision_at_k([3], k=-1)
        with self.assertRaises(ValueError):
            recall_at_k([3], total_relevant=-1)


class TraditionalAnswerMetricTests(unittest.TestCase):
    def test_exact_rouge_scores(self) -> None:
        text = "Apple reported strong quarterly earnings"
        scores = rouge_scores(text, text)

        for metric in ("rouge_1", "rouge_2", "rouge_l"):
            self.assertEqual(scores[metric]["precision"], 1.0)
            self.assertEqual(scores[metric]["recall"], 1.0)
            self.assertEqual(scores[metric]["f1"], 1.0)

    def test_pdf_rouge_examples(self) -> None:
        reference = "Apple reported strong quarterly earnings"
        candidate = "Apple announced strong quarterly results"

        rouge_1 = rouge_n(reference, candidate, n=1)
        rouge_2 = rouge_n(reference, candidate, n=2)
        self.assertAlmostEqual(rouge_1["f1"], 0.6)
        self.assertAlmostEqual(rouge_2["f1"], 0.25)
        self.assertGreater(rouge_l(reference, candidate)["f1"], 0.0)

    def test_empty_rouge_does_not_reward_empty_answer(self) -> None:
        self.assertEqual(rouge_n("", "")["f1"], 0.0)
        self.assertEqual(rouge_l("", "")["f1"], 0.0)

    def test_exact_and_partial_fuzzy_matching(self) -> None:
        self.assertEqual(fuzzy_ratio("Revenue was $85.8 billion", "Revenue was $85.8 billion"), 100.0)
        self.assertEqual(
            fuzzy_partial_ratio(
                "Revenue was $85.8 billion",
                "In Q3 2024, Apple's revenue was $85.8 billion, up 5% YoY.",
            ),
            100.0,
        )

    def test_pdf_levenshtein_example(self) -> None:
        self.assertEqual(levenshtein_distance("Apple Inc", "Apple Inc."), 1)
        self.assertEqual(fuzzy_ratio("Apple Inc", "Apple Inc."), 90.0)

    def test_cosine_similarity(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(cosine_similarity([1, 1], [-1, -1]), -1.0)
        self.assertEqual(cosine_similarity([], []), 0.0)
        self.assertEqual(cosine_similarity([0, 0], [1, 1]), 0.0)
        with self.assertRaises(ValueError):
            cosine_similarity([1], [1, 2])
        with self.assertRaises(ValueError):
            cosine_similarity([math.nan], [1])


class JudgeNormalizationTests(unittest.TestCase):
    FENCED_RESPONSE = """
    Here is the evaluation:
    ```json
    {
      "search": {
        "documents": [
          {"rank": 1, "source_id": "S1", "relevance_grade": 3},
          {"rank": 2, "source_id": "S2", "relevance_grade": 1},
          {"rank": 3, "source_id": "S3", "relevance_grade": 2}
        ],
        "rewritten_query": "Apple Q3 2024 revenue"
      },
      "answer": {
        "source_support": [
          {"source_id": "S1", "useful": true},
          {"source_id": "S2", "useful": false},
          {"source_id": "S3", "useful": true}
        ],
        "claims": [
          {"claim": "Revenue was $85.8B", "classification": "TP"},
          {"claim": "Mac sales fell 3%", "classification": "FP"},
          {"claim": "Revenue grew 5%", "classification": "FN"}
        ],
        "noncommittal": false
      }
    }
    ```
    """

    def test_extracts_fenced_json_surrounded_by_prose(self) -> None:
        extracted = extract_json_object(self.FENCED_RESPONSE)
        self.assertIn("search", extracted)
        self.assertIn("answer", extracted)

    def test_normalizes_combined_judgment(self) -> None:
        normalized = normalize_combined_judgment(self.FENCED_RESPONSE)
        self.assertEqual(search_grades_from_judgment(self.FENCED_RESPONSE), [3.0, 1.0, 2.0])
        self.assertEqual(normalized["search"]["rewritten_query"], "Apple Q3 2024 revenue")
        self.assertEqual(normalized["answer"]["relevancy"], 1)

    def test_derives_answer_metrics_from_labels(self) -> None:
        metrics = derive_answer_metrics(self.FENCED_RESPONSE)
        self.assertAlmostEqual(metrics["grounding_percent"], 200 / 3)
        self.assertAlmostEqual(metrics["tp_percent"], 50.0)
        self.assertAlmostEqual(metrics["fp_percent"], 50.0)
        self.assertAlmostEqual(metrics["supported_claims_percent"], 50.0)
        self.assertAlmostEqual(metrics["hallucinated_claims_percent"], 50.0)
        self.assertAlmostEqual(metrics["claim_completeness_percent"], 50.0)
        self.assertEqual(metrics["relevancy"], 1)
        self.assertEqual(metrics["claim_counts"], {"tp": 1, "fp": 1, "fn": 1})

    def test_evaluates_combined_judgment(self) -> None:
        evaluation = evaluate_combined_judgment(self.FENCED_RESPONSE, total_relevant=2)
        self.assertAlmostEqual(evaluation["search"]["precision_at_k"], 2 / 3)
        self.assertAlmostEqual(evaluation["search"]["recall_at_k"], 1.0)
        self.assertEqual(evaluation["answer"]["relevancy"], 1)

    def test_accepts_common_variant_shapes(self) -> None:
        raw = {
            "retrieval_evaluation": {
                "relevance_grades": ["highly relevant", "not relevant", True]
            },
            "generation_evaluation": {
                "sources": {"doc-a": "supported", "doc-b": "not supported"},
                "supported_claims": ["Claim A", "Claim B"],
                "unsupported_claims": ["Claim C"],
                "missing_claims": ["Claim D"],
                "relevance": "yes",
            },
        }
        normalized = normalize_combined_judgment(raw)
        metrics = derive_answer_metrics(raw)

        self.assertEqual(search_grades_from_judgment(raw), [3.0, 0.0, 3.0])
        self.assertEqual(normalized["answer"]["claim_counts"], {"tp": 2, "fp": 1, "fn": 1})
        self.assertAlmostEqual(metrics["grounding_percent"], 50.0)
        self.assertAlmostEqual(metrics["tp_percent"], 200 / 3)
        self.assertAlmostEqual(metrics["fp_percent"], 100 / 3)
        self.assertAlmostEqual(metrics["claim_completeness_percent"], 200 / 3)

    def test_accepts_live_judge_prompt_shapes(self) -> None:
        search_raw = {
            "items": [
                {"source_id": "S1", "grade": 3, "reason": "direct"},
                {"source_id": "S2", "grade": 0, "reason": "unrelated"},
            ],
            "rewritten_query": "clearer query",
            "answerable_from_sources": True,
        }
        answer_raw = {
            "answerable_from_sources": True,
            "sources": [{"source_id": "S1", "useful": True}],
            "claims": [{"claim": "Supported fact", "label": "TP"}],
            "missing_key_facts": ["Omitted fact"],
            "noncommittal": False,
            "relevant": True,
        }

        search = normalize_combined_judgment(search_raw)["search"]
        answer = normalize_combined_judgment(answer_raw)["answer"]
        self.assertEqual(search_grades_from_judgment(search_raw), [3.0, 0.0])
        self.assertTrue(search["answerable_from_sources"])
        self.assertTrue(answer["answerable_from_sources"])
        self.assertEqual(answer["claim_counts"], {"tp": 1, "fp": 0, "fn": 1})

    def test_malformed_json_is_safe(self) -> None:
        self.assertEqual(extract_json_object("```json\n{not valid json}\n```"), {})
        normalized = normalize_combined_judgment("not JSON at all")
        self.assertEqual(normalized["search"]["documents"], [])
        self.assertEqual(normalized["answer"]["claims"], [])

        metrics = derive_answer_metrics("not JSON at all")
        self.assertEqual(metrics["grounding_percent"], 0.0)
        self.assertEqual(metrics["tp_percent"], 0.0)
        self.assertEqual(metrics["fp_percent"], 0.0)
        self.assertEqual(metrics["claim_completeness_percent"], 0.0)
        self.assertEqual(metrics["relevancy"], 0)

    def test_count_and_reported_percentage_fallbacks(self) -> None:
        count_only = {
            "answer": {
                "tp_count": 2,
                "fp_count": 1,
                "fn_count": 1,
                "grounding_percent": 0.75,
                "relevant": True,
            }
        }
        metrics = derive_answer_metrics(count_only)
        self.assertAlmostEqual(metrics["grounding_percent"], 75.0)
        self.assertAlmostEqual(metrics["tp_percent"], 200 / 3)
        self.assertAlmostEqual(metrics["fp_percent"], 100 / 3)
        self.assertAlmostEqual(metrics["claim_completeness_percent"], 200 / 3)

        reported_only = {
            "answer": {
                "grounding_percent": 80,
                "tp_percent": 90,
                "fp_percent": 10,
                "claim_completeness_percent": 85,
            }
        }
        combined = evaluate_combined_judgment(reported_only)
        self.assertEqual(combined["answer"]["grounding_percent"], 80.0)
        self.assertEqual(combined["answer"]["tp_percent"], 90.0)
        self.assertEqual(combined["answer"]["fp_percent"], 10.0)
        self.assertEqual(combined["answer"]["claim_completeness_percent"], 85.0)

    def test_extracts_python_style_dict_as_last_resort(self) -> None:
        result = extract_json_object("{'search': {'grades': [3, 0]}}")
        self.assertEqual(result["search"]["grades"], [3, 0])


if __name__ == "__main__":
    unittest.main()
