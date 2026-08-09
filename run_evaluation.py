#!/usr/bin/env python3
"""Run the labeled fitness RAG benchmark and aggregate available metrics.

The command intentionally imports ``RAGAgent`` only after CLI parsing and data
validation.  As a result, ``--help`` and the loader/aggregation helpers work
without initializing embeddings, FAISS, or an LLM client.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import fmean
import sys
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK_PATH = PROJECT_ROOT / "data" / "evaluation_questions.json"
REQUIRED_FIELDS = {
    "id",
    "question",
    "reference_answer",
    "answerable",
    "relevant_documents",
}


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the labeled fitness RAG benchmark. Metrics that are unavailable "
            "are omitted from aggregate means rather than counted as zero."
        )
    )
    parser.add_argument(
        "--limit",
        type=_positive_integer,
        metavar="N",
        help="evaluate only the first N benchmark questions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="PATH",
        help="write the complete run and aggregates as JSON",
    )
    return parser.parse_args(argv)


def load_benchmark(path: str | Path = BENCHMARK_PATH) -> list[dict[str, Any]]:
    """Load and validate the small page-labeled benchmark using only stdlib."""
    benchmark_path = Path(path)
    with benchmark_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        raise ValueError("benchmark root must be a JSON list")

    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"benchmark record {index} must be an object")
        missing = REQUIRED_FIELDS.difference(record)
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"benchmark record {index} is missing: {fields}")

        record_id = record["id"]
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"benchmark record {index} has an invalid id")
        if record_id in seen_ids:
            raise ValueError(f"duplicate benchmark id: {record_id}")
        seen_ids.add(record_id)

        for field in ("question", "reference_answer"):
            if not isinstance(record[field], str) or not record[field].strip():
                raise ValueError(f"benchmark record {record_id!r} has an invalid {field}")
        if not isinstance(record["answerable"], bool):
            raise ValueError(f"benchmark record {record_id!r} has a non-boolean answerable")
        if not isinstance(record["relevant_documents"], list):
            raise ValueError(
                f"benchmark record {record_id!r} has invalid relevant_documents"
            )

        for document_index, document in enumerate(record["relevant_documents"], start=1):
            if not isinstance(document, dict):
                raise ValueError(
                    f"benchmark record {record_id!r} relevant document "
                    f"{document_index} must be an object"
                )
            if not isinstance(document.get("source"), str) or not document["source"].strip():
                raise ValueError(
                    f"benchmark record {record_id!r} relevant document "
                    f"{document_index} has an invalid source"
                )
            page = document.get("page")
            if isinstance(page, bool) or not isinstance(page, int) or page < 1:
                raise ValueError(
                    f"benchmark record {record_id!r} relevant document "
                    f"{document_index} has an invalid page"
                )

        validated.append(dict(record))
    return validated


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _put_number(target: dict[str, float], name: str, value: Any) -> None:
    number = _finite_number(value)
    if number is not None:
        target[name] = number


def extract_case_metrics(
    record: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, float]:
    """Flatten the public ``RAGAgent.query`` evaluation payload."""
    metrics: dict[str, float] = {}
    evaluation = _mapping(result.get("evaluation"))
    search_metrics = _mapping(_nested(evaluation, "search", "metrics"))
    answer_metrics = _mapping(_nested(evaluation, "answer", "metrics"))
    reference_metrics = _mapping(_nested(evaluation, "answer", "reference_metrics"))

    _put_number(metrics, "reciprocal_rank", search_metrics.get("reciprocal_rank"))
    _put_number(
        metrics,
        "average_precision_at_k",
        search_metrics.get("average_precision_at_k"),
    )
    _put_number(metrics, "precision_at_k", search_metrics.get("precision_at_k"))
    _put_number(metrics, "recall_at_k", search_metrics.get("recall_at_k"))
    _put_number(metrics, "f1", search_metrics.get("f1"))
    _put_number(metrics, "ndcg_at_k", search_metrics.get("ndcg_at_k"))

    for source_name, output_name in (
        ("rouge_1", "rouge_1_f1"),
        ("rouge_2", "rouge_2_f1"),
        ("rouge_l", "rouge_l_f1"),
    ):
        score = _mapping(reference_metrics.get(source_name)).get("f1")
        _put_number(metrics, output_name, score)
    _put_number(metrics, "fuzzy_ratio", reference_metrics.get("fuzzy_ratio"))
    _put_number(
        metrics,
        "fuzzy_partial_ratio",
        reference_metrics.get("fuzzy_partial_ratio"),
    )
    _put_number(
        metrics,
        "semantic_similarity",
        reference_metrics.get("semantic_similarity"),
    )

    _put_number(metrics, "grounding_percent", answer_metrics.get("grounding_percent"))
    _put_number(
        metrics,
        "tp_percent",
        answer_metrics.get("supported_claims_percent"),
    )
    _put_number(
        metrics,
        "fp_percent",
        answer_metrics.get("hallucinated_claims_percent"),
    )
    _put_number(metrics, "relevancy", answer_metrics.get("relevancy"))

    if record.get("answerable") is False:
        answer_evaluation = _mapping(evaluation.get("answer"))
        correct_abstention = answer_evaluation.get("correct_abstention")
        if isinstance(correct_abstention, bool):
            metrics["correct_abstention"] = float(correct_abstention)
        elif (
            answer_evaluation.get("mode") == "abstention_check"
            and isinstance(answer_evaluation.get("passed"), bool)
        ):
            metrics["correct_abstention"] = float(answer_evaluation["passed"])

    return metrics


ANSWERABLE_AGGREGATES = (
    ("reciprocal_rank", "mrr"),
    ("average_precision_at_k", "map"),
    ("precision_at_k", "mean_precision_at_k"),
    ("recall_at_k", "mean_recall_at_k"),
    ("f1", "mean_f1"),
    ("ndcg_at_k", "mean_ndcg_at_k"),
    ("rouge_1_f1", "mean_rouge_1_f1"),
    ("rouge_2_f1", "mean_rouge_2_f1"),
    ("rouge_l_f1", "mean_rouge_l_f1"),
    ("fuzzy_ratio", "mean_fuzzy_ratio"),
    ("fuzzy_partial_ratio", "mean_fuzzy_partial_ratio"),
    ("semantic_similarity", "mean_semantic_similarity"),
    ("grounding_percent", "mean_grounding_percent"),
    ("tp_percent", "mean_tp_percent"),
    ("fp_percent", "mean_fp_percent"),
    ("relevancy", "mean_relevancy"),
)


def aggregate_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Macro-average available per-question values without imputing N/A as zero."""
    answerable = [case for case in cases if case.get("answerable") is True]
    unanswerable = [case for case in cases if case.get("answerable") is False]
    completed = [case for case in cases if "error" not in case]

    answerable_metrics: dict[str, float] = {}
    sample_counts: dict[str, int] = {}
    for case_key, aggregate_key in ANSWERABLE_AGGREGATES:
        values = [
            number
            for case in answerable
            if (
                number := _finite_number(
                    _mapping(case.get("metrics")).get(case_key)
                )
            )
            is not None
        ]
        if values:
            answerable_metrics[aggregate_key] = fmean(values)
            sample_counts[aggregate_key] = len(values)

    abstentions = [
        number
        for case in unanswerable
        if (
            number := _finite_number(
                _mapping(case.get("metrics")).get("correct_abstention")
            )
        )
        is not None
    ]

    aggregate: dict[str, Any] = {
        "counts": {
            "selected": len(cases),
            "completed": len(completed),
            "errors": len(cases) - len(completed),
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
        },
        "answerable_metrics": answerable_metrics,
        "metric_sample_counts": sample_counts,
    }
    if abstentions:
        aggregate["unanswerable_metrics"] = {
            "abstention_accuracy": fmean(abstentions),
            "correct_abstentions": int(sum(abstentions)),
            "evaluated": len(abstentions),
        }
    return aggregate


def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    query: Callable[..., Mapping[str, Any]],
    on_case: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate records through an injected query callable.

    Keeping this loop independent of ``RAGAgent`` makes loader and aggregation
    behavior testable without model downloads or API calls.
    """
    cases: list[dict[str, Any]] = []
    for record in records:
        case: dict[str, Any] = {
            "id": record["id"],
            "question": record["question"],
            "reference_answer": record["reference_answer"],
            "answerable": record["answerable"],
            "relevant_documents": record["relevant_documents"],
        }
        try:
            result = query(
                record["question"],
                evaluate=True,
                reference_answer=record["reference_answer"],
                relevant_documents=record["relevant_documents"],
                answerable=record["answerable"],
            )
            if not isinstance(result, Mapping):
                raise TypeError("RAGAgent.query must return a mapping")
            case["result"] = dict(result)
            case["metrics"] = extract_case_metrics(record, result)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # Continue so one failed request does not erase the run.
            # Deliberately avoid serializing exception text: SDK exceptions can
            # contain request metadata that should not be copied to reports.
            case["error"] = {"type": type(exc).__name__}
            case["metrics"] = {}
        cases.append(case)
        if on_case is not None:
            on_case(case)
    return cases


def _status(evaluation: Mapping[str, Any]) -> str:
    if evaluation.get("status") == "error":
        return "ERR"
    passed = evaluation.get("passed")
    if passed is True:
        return "PASS"
    if passed is False:
        return "FAIL"
    return "N/A"


def _overall_status(case: Mapping[str, Any]) -> str:
    if "error" in case:
        return "ERROR"
    result = _mapping(case.get("result"))
    status = _nested(_mapping(result.get("evaluation")), "overall", "status")
    return {
        "passed": "PASS",
        "needs_review": "REVIEW",
        "evaluator_error": "EVAL_ERR",
        "disabled": "DISABLED",
    }.get(str(status), str(status or "N/A").upper())


def _decimal(value: Any) -> str:
    number = _finite_number(value)
    return f"{number:.3f}" if number is not None else "-"


def _percent(value: Any) -> str:
    number = _finite_number(value)
    return f"{number:.1f}%" if number is not None else "-"


TABLE_HEADERS = (
    "ID",
    "TYPE",
    "SEARCH",
    "ANSWER",
    "RR",
    "AP",
    "R-L",
    "GROUND",
    "FP",
    "ABST",
    "RETRIES",
    "STATUS",
)


def _table_row(case: Mapping[str, Any]) -> tuple[str, ...]:
    metrics = _mapping(case.get("metrics"))
    result = _mapping(case.get("result"))
    evaluation = _mapping(result.get("evaluation"))
    search = _mapping(evaluation.get("search"))
    answer = _mapping(evaluation.get("answer"))
    overall = _mapping(evaluation.get("overall"))
    answerable = case.get("answerable") is True

    abstention = "-"
    if not answerable:
        correct = _finite_number(metrics.get("correct_abstention"))
        abstention = "YES" if correct == 1.0 else "NO" if correct == 0.0 else "N/A"

    return (
        str(case.get("id", "?")),
        "A" if answerable else "U",
        _status(search) if "error" not in case else "ERR",
        _status(answer) if "error" not in case else "ERR",
        _decimal(metrics.get("reciprocal_rank")) if answerable else "-",
        _decimal(metrics.get("average_precision_at_k")) if answerable else "-",
        _decimal(metrics.get("rouge_l_f1")) if answerable else "-",
        _percent(metrics.get("grounding_percent")) if answerable else "-",
        _percent(metrics.get("fp_percent")) if answerable else "-",
        abstention,
        (
            f"S{int(overall.get('search_retries', 0))}/"
            f"A{int(overall.get('answer_retries', 0))}"
            if overall
            else "-"
        ),
        _overall_status(case),
    )


class CompactTable:
    """Print fixed-width rows as each potentially slow question completes."""

    def __init__(self, records: Sequence[Mapping[str, Any]]):
        id_width = max([len(TABLE_HEADERS[0]), *(len(str(row["id"])) for row in records)])
        self.widths = (id_width, 4, 6, 6, 5, 5, 5, 7, 7, 5, 7, 8)
        self._print(TABLE_HEADERS)
        print("  ".join("-" * width for width in self.widths))

    def _print(self, values: Sequence[str]) -> None:
        print(
            "  ".join(
                str(value)[:width].ljust(width)
                for value, width in zip(values, self.widths)
            ),
            flush=True,
        )

    def add(self, case: Mapping[str, Any]) -> None:
        self._print(_table_row(case))


METRIC_LABELS = (
    ("mrr", "MRR"),
    ("map", "MAP"),
    ("mean_precision_at_k", "Mean Precision@k"),
    ("mean_recall_at_k", "Mean Recall@k"),
    ("mean_f1", "Mean retrieval F1"),
    ("mean_ndcg_at_k", "Mean NDCG@k"),
    ("mean_rouge_1_f1", "Mean ROUGE-1 F1"),
    ("mean_rouge_2_f1", "Mean ROUGE-2 F1"),
    ("mean_rouge_l_f1", "Mean ROUGE-L F1"),
    ("mean_fuzzy_ratio", "Mean fuzzy ratio"),
    ("mean_fuzzy_partial_ratio", "Mean fuzzy partial ratio"),
    ("mean_semantic_similarity", "Mean semantic similarity"),
    ("mean_grounding_percent", "Mean judge grounding"),
    ("mean_tp_percent", "Mean judge TP"),
    ("mean_fp_percent", "Mean judge FP"),
    ("mean_relevancy", "Mean judge relevancy"),
)


def print_aggregate(aggregate: Mapping[str, Any]) -> None:
    metrics = _mapping(aggregate.get("answerable_metrics"))
    counts = _mapping(aggregate.get("metric_sample_counts"))
    selected_counts = _mapping(aggregate.get("counts"))

    print("\nAnswerable-query macro means (undefined values excluded):")
    if not metrics:
        print("  No answerable metrics were available.")
    for key, label in METRIC_LABELS:
        value = _finite_number(metrics.get(key))
        if value is None:
            continue
        suffix = "%" if key in {
            "mean_grounding_percent",
            "mean_tp_percent",
            "mean_fp_percent",
        } else ""
        precision = 1 if suffix or "fuzzy" in key else 3
        rendered = f"{value:.{precision}f}{suffix}"
        print(f"  {label:<29} {rendered:>9}  (n={counts.get(key, 0)})")

    unanswerable = _mapping(aggregate.get("unanswerable_metrics"))
    accuracy = _finite_number(unanswerable.get("abstention_accuracy"))
    if accuracy is not None:
        print(
            "\nUnanswerable abstention accuracy: "
            f"{accuracy:.3f} "
            f"({unanswerable.get('correct_abstentions', 0)}/"
            f"{unanswerable.get('evaluated', 0)})"
        )
    elif selected_counts.get("unanswerable", 0):
        print("\nUnanswerable abstention accuracy: unavailable")

    print(
        "\nCompleted "
        f"{selected_counts.get('completed', 0)}/{selected_counts.get('selected', 0)} "
        f"questions; errors={selected_counts.get('errors', 0)}."
    )


def _write_json(
    path: Path,
    cases: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    limit: int | None,
) -> None:
    output_path = path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": str(BENCHMARK_PATH.relative_to(PROJECT_ROOT)),
        "limit": limit,
        "aggregate": aggregate,
        "cases": list(cases),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        handle.write("\n")
    print(f"JSON report written to {output_path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        records = load_benchmark()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Cannot load benchmark: {exc}", file=sys.stderr)
        return 2

    if args.limit is not None:
        records = records[: args.limit]

    # Keep --help and helper imports lightweight. Initializing RAGAgent may load
    # a local embedding model and requires the configured LLM credentials.
    try:
        from app.agent import RAGAgent

        agent = RAGAgent()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(
            f"Cannot initialize RAGAgent ({type(exc).__name__}). "
            "Check the vector store, dependencies, and configured credentials.",
            file=sys.stderr,
        )
        return 2

    print(f"Benchmark: {BENCHMARK_PATH.relative_to(PROJECT_ROOT)} ({len(records)} questions)\n")
    table = CompactTable(records)
    try:
        cases = evaluate_records(records, agent.query, on_case=table.add)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted.", file=sys.stderr)
        return 130

    aggregate = aggregate_cases(cases)
    print_aggregate(aggregate)

    if args.output is not None:
        try:
            _write_json(args.output, cases, aggregate, args.limit)
        except (OSError, TypeError, ValueError) as exc:
            print(f"Cannot write JSON report: {exc}", file=sys.stderr)
            return 2

    return 1 if aggregate["counts"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
