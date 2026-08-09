"""Deterministic evaluation helpers for the RAG pipeline.

The module deliberately depends only on the Python standard library.  It
contains two kinds of helpers:

* metric implementations, which are deterministic and straightforward to
  unit test; and
* normalization helpers for the JSON returned by an LLM judge.  The judge is
  responsible only for labels (document relevance, source usefulness, and
  claim classifications); percentages are calculated here.

Search functions accept relevance grades in the inclusive range 0--3.  Unless
explicitly stated otherwise, a grade of 2 or 3 is considered relevant.
"""

from __future__ import annotations

import ast
from collections import Counter
import json
import math
import re
from typing import Any, Mapping, Sequence


DEFAULT_RELEVANCE_THRESHOLD = 2.0


# ---------------------------------------------------------------------------
# Search metrics
# ---------------------------------------------------------------------------


def _normalise_grade(value: Any) -> float:
    """Convert a judge grade to a finite number clamped to 0--3."""

    if isinstance(value, bool):
        return 3.0 if value else 0.0

    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        cleaned = value.strip().casefold()
        if cleaned in {"relevant", "highly relevant", "yes", "true"}:
            return 3.0
        if cleaned in {"partially relevant", "partial", "somewhat relevant"}:
            return 1.0
        if cleaned in {"irrelevant", "not relevant", "no", "false"}:
            return 0.0
        match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
        if not match:
            return 0.0
        number = float(match.group())
    else:
        return 0.0

    if not math.isfinite(number):
        return 0.0
    return min(3.0, max(0.0, number))


def normalize_relevance_grades(grades: Sequence[Any] | None) -> list[float]:
    """Return safe 0--3 grades while preserving the supplied ranking order."""

    if not grades:
        return []
    return [_normalise_grade(grade) for grade in grades]


def _top_k(values: Sequence[Any], k: int | None) -> list[Any]:
    if k is None:
        return list(values)
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer or None")
    if k < 0:
        raise ValueError("k cannot be negative")
    return list(values[:k])


def binary_relevance(
    grades: Sequence[Any] | None,
    *,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> list[int]:
    """Convert 0--3 grades into binary labels using ``grade >= threshold``."""

    return [int(grade >= threshold) for grade in normalize_relevance_grades(grades)]


def precision_at_k(
    grades: Sequence[Any] | None,
    k: int | None = None,
    *,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> float:
    """Return the fraction of returned top-k results that are relevant.

    When fewer than ``k`` results exist, the denominator is the number actually
    returned.  Empty input therefore has precision 0 rather than a division
    error.
    """

    labels = _top_k(binary_relevance(grades, threshold=threshold), k)
    return sum(labels) / len(labels) if labels else 0.0


def reciprocal_rank(
    grades: Sequence[Any] | None,
    k: int | None = None,
    *,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> float:
    """Return the reciprocal rank of the first relevant top-k result."""

    labels = _top_k(binary_relevance(grades, threshold=threshold), k)
    for rank, relevant in enumerate(labels, start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(
    grades: Sequence[Any] | None,
    k: int | None = None,
    *,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> float:
    """Return average precision across relevant positions in the observed top-k.

    This is the per-query value averaged to obtain MAP across a benchmark.  In
    the absence of corpus-wide labels, its denominator is the number of
    relevant results observed in the supplied ranking, matching the worked
    example in the evaluation lesson.
    """

    labels = _top_k(binary_relevance(grades, threshold=threshold), k)
    relevant_seen = 0
    precision_sum = 0.0

    for rank, relevant in enumerate(labels, start=1):
        if relevant:
            relevant_seen += 1
            precision_sum += relevant_seen / rank

    return precision_sum / relevant_seen if relevant_seen else 0.0


def ndcg_at_k(
    grades: Sequence[Any] | None,
    k: int | None = None,
    *,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    graded: bool = False,
) -> float:
    """Return normalized discounted cumulative gain for the top-k ranking.

    By default grades are converted to binary relevance to reproduce the PDF's
    worked example.  Set ``graded=True`` to use the standard exponential gain
    ``2**grade - 1`` and preserve the distinction between grades 0--3.
    """

    normalised = _top_k(normalize_relevance_grades(grades), k)
    if not normalised:
        return 0.0

    if graded:
        gains = [(2.0**grade) - 1.0 for grade in normalised]
    else:
        gains = [float(grade >= threshold) for grade in normalised]

    def discounted_gain(ranked_gains: Sequence[float]) -> float:
        return sum(
            gain / math.log2(rank + 1)
            for rank, gain in enumerate(ranked_gains, start=1)
        )

    actual = discounted_gain(gains)
    ideal = discounted_gain(sorted(gains, reverse=True))
    return actual / ideal if ideal else 0.0


def recall_at_k(
    grades: Sequence[Any] | None,
    total_relevant: int,
    k: int | None = None,
    *,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> float | None:
    """Return recall when the corpus-wide relevant-document count is known.

    Recall is undefined when ``total_relevant`` is zero, so ``None`` is
    returned.  Negative totals are rejected because they indicate bad ground
    truth data.
    """

    if isinstance(total_relevant, bool) or not isinstance(total_relevant, int):
        raise TypeError("total_relevant must be an integer")
    if total_relevant < 0:
        raise ValueError("total_relevant cannot be negative")
    if total_relevant == 0:
        return None

    labels = _top_k(binary_relevance(grades, threshold=threshold), k)
    relevant_retrieved = min(sum(labels), total_relevant)
    return relevant_retrieved / total_relevant


def f1_score(precision: float, recall: float) -> float:
    """Return the harmonic mean of precision and recall, guarded at zero."""

    denominator = precision + recall
    return (2.0 * precision * recall / denominator) if denominator else 0.0


def evaluate_search_metrics(
    grades: Sequence[Any] | None,
    k: int | None = None,
    *,
    total_relevant: int | None = None,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    graded_ndcg: bool = False,
) -> dict[str, Any]:
    """Calculate the complete deterministic search-evaluation summary."""

    normalised = _top_k(normalize_relevance_grades(grades), k)
    labels = binary_relevance(normalised, threshold=threshold)
    precision = precision_at_k(normalised, threshold=threshold)

    recall: float | None = None
    f1: float | None = None
    if total_relevant is not None:
        recall = recall_at_k(
            normalised,
            total_relevant,
            threshold=threshold,
        )
        if recall is not None:
            f1 = f1_score(precision, recall)

    return {
        "grades": normalised,
        "retrieved_count": len(normalised),
        "relevant_retrieved": sum(labels),
        "precision_at_k": precision,
        "reciprocal_rank": reciprocal_rank(normalised, threshold=threshold),
        "average_precision_at_k": average_precision_at_k(
            normalised,
            threshold=threshold,
        ),
        "ndcg_at_k": ndcg_at_k(
            normalised,
            threshold=threshold,
            graded=graded_ndcg,
        ),
        "recall_at_k": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Traditional answer metrics
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokens(text: str | None) -> list[str]:
    return _TOKEN_RE.findall((text or "").casefold())


def _prf(overlap: int, candidate_count: int, reference_count: int) -> dict[str, float]:
    precision = overlap / candidate_count if candidate_count else 0.0
    recall = overlap / reference_count if reference_count else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
    }


def rouge_n(reference: str | None, candidate: str | None, n: int = 1) -> dict[str, float]:
    """Return count-aware ROUGE-N precision, recall, and F1."""

    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n <= 0:
        raise ValueError("n must be greater than zero")

    def ngrams(tokens: Sequence[str]) -> list[tuple[str, ...]]:
        return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]

    reference_ngrams = Counter(ngrams(_tokens(reference)))
    candidate_ngrams = Counter(ngrams(_tokens(candidate)))
    overlap = sum((reference_ngrams & candidate_ngrams).values())
    return _prf(overlap, sum(candidate_ngrams.values()), sum(reference_ngrams.values()))


def rouge_l(reference: str | None, candidate: str | None) -> dict[str, float]:
    """Return ROUGE-L using a token-level longest common subsequence."""

    reference_tokens = _tokens(reference)
    candidate_tokens = _tokens(candidate)
    if not reference_tokens or not candidate_tokens:
        return _prf(0, len(candidate_tokens), len(reference_tokens))

    # A one-row dynamic program keeps memory linear in the candidate length.
    previous = [0] * (len(candidate_tokens) + 1)
    for reference_token in reference_tokens:
        current = [0]
        for index, candidate_token in enumerate(candidate_tokens, start=1):
            if reference_token == candidate_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current

    return _prf(previous[-1], len(candidate_tokens), len(reference_tokens))


def rouge_scores(reference: str | None, candidate: str | None) -> dict[str, dict[str, float]]:
    """Return ROUGE-1, ROUGE-2, and ROUGE-L in one JSON-friendly mapping."""

    return {
        "rouge_1": rouge_n(reference, candidate, n=1),
        "rouge_2": rouge_n(reference, candidate, n=2),
        "rouge_l": rouge_l(reference, candidate),
    }


def levenshtein_distance(left: str, right: str) -> int:
    """Return the minimum insertions, deletions, and substitutions required."""

    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    # Put the shorter string on the columns to keep memory use bounded.
    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            insertion = current[column - 1] + 1
            deletion = previous[column] + 1
            substitution = previous[column - 1] + (left_char != right_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def _normalise_fuzzy_text(value: str | None) -> str:
    # Preserve punctuation (it represents a real edit) while making casing and
    # repeated whitespace irrelevant for answer comparison.
    return " ".join((value or "").casefold().split())


def _fuzzy_ratio_normalised(left: str, right: str) -> float:
    if not left and not right:
        return 100.0
    longest = max(len(left), len(right))
    if not longest:
        return 100.0
    distance = levenshtein_distance(left, right)
    return max(0.0, 100.0 * (1.0 - (distance / longest)))


def fuzzy_ratio(left: str | None, right: str | None) -> float:
    """Return case-insensitive normalized Levenshtein similarity from 0--100."""

    return _fuzzy_ratio_normalised(
        _normalise_fuzzy_text(left),
        _normalise_fuzzy_text(right),
    )


def fuzzy_partial_ratio(left: str | None, right: str | None) -> float:
    """Return the best fuzzy score for the shorter text within the longer one."""

    first = _normalise_fuzzy_text(left)
    second = _normalise_fuzzy_text(right)
    if len(first) > len(second):
        first, second = second, first
    if not first:
        return 100.0 if not second else 0.0
    if first in second:
        return 100.0

    window_size = len(first)
    return max(
        _fuzzy_ratio_normalised(first, second[start : start + window_size])
        for start in range(len(second) - window_size + 1)
    )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for equal-length vectors without NumPy.

    Empty vectors and zero-norm vectors return 0.  A dimension mismatch is a
    programming error and raises ``ValueError``.
    """

    if len(left) != len(right):
        raise ValueError("vectors must have the same dimensions")
    if not left:
        return 0.0

    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if not all(math.isfinite(value) for value in left_values + right_values):
        raise ValueError("vectors must contain only finite numbers")

    dot_product = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if not left_norm or not right_norm:
        return 0.0

    similarity = dot_product / (left_norm * right_norm)
    return min(1.0, max(-1.0, similarity))


# ---------------------------------------------------------------------------
# Combined LLM-judge JSON normalization
# ---------------------------------------------------------------------------


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)


def _message_text(raw: Any) -> str:
    """Extract text from common chat-message and content-block representations."""

    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        return raw

    content = getattr(raw, "content", None)
    if content is not None and content is not raw:
        return _message_text(content)

    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        parts: list[str] = []
        for block in raw:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)

    return ""


def _parse_mapping(candidate: str) -> dict[str, Any]:
    candidate = candidate.strip().lstrip("\ufeff")
    if not candidate:
        return {}

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    except (json.JSONDecodeError, TypeError):
        pass

    # raw_decode can recover an object surrounded by prose.
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", candidate):
        try:
            parsed, _ = decoder.raw_decode(candidate[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            return dict(parsed)

    # LLMs occasionally emit a Python-style dict. literal_eval is safe and
    # gives a useful last chance without attempting arbitrary code execution.
    try:
        parsed = ast.literal_eval(candidate)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    except (SyntaxError, ValueError, TypeError):
        pass
    return {}


def extract_json_object(raw: Any) -> dict[str, Any]:
    """Extract the first JSON-like object from a judge response.

    Accepted inputs include dictionaries, Pydantic-like objects, chat messages,
    content-block lists, fenced JSON, and JSON surrounded by prose.  Malformed
    input safely returns an empty dictionary.
    """

    if isinstance(raw, Mapping):
        return dict(raw)

    for method_name in ("model_dump", "dict"):
        method = getattr(raw, method_name, None)
        if callable(method):
            try:
                value = method()
            except (TypeError, ValueError):
                continue
            if isinstance(value, Mapping):
                return dict(value)

    text = _message_text(raw)
    if not text:
        return {}

    fenced_candidates = _FENCED_JSON_RE.findall(text)
    for candidate in [text, *fenced_candidates]:
        parsed = _parse_mapping(candidate)
        if parsed:
            return parsed
    return {}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first_value(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, str):
        match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
        if match:
            result = float(match.group())
            return result if math.isfinite(result) else None
    if isinstance(value, Mapping):
        return _number(_first_value(value, ("score", "value", "rating", "percent")))
    return None


def _count(value: Any) -> int | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value)
    number = _number(value)
    if number is None:
        return None
    return max(0, int(number))


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value >= 0.5
    if isinstance(value, Mapping):
        nested = _first_value(value, ("value", "score", "supported", "useful", "relevant"))
        return _as_bool(nested, default=default)
    if isinstance(value, str):
        cleaned = value.strip().casefold().replace("_", " ").replace("-", " ")
        negative = (
            "not supported",
            "unsupported",
            "not useful",
            "irrelevant",
            "false",
            "no",
            "hallucinated",
        )
        if cleaned in negative or any(cleaned.startswith(item + " ") for item in negative):
            return False
        positive = {
            "supported",
            "useful",
            "relevant",
            "true",
            "yes",
            "grounded",
            "direct",
            "helpful",
        }
        if cleaned in positive:
            return True
        number = _number(cleaned)
        if number is not None:
            return number >= 0.5
    return default


def binary_relevancy(value: Any, *, noncommittal: Any = None) -> int:
    """Normalize judge relevancy to 0/1, optionally inverting noncommittal."""

    if value is not None:
        return int(_as_bool(value))
    if noncommittal is not None:
        return int(not _as_bool(noncommittal))
    return 0


def _normalise_claim_label(value: Any, supported: Any = None) -> str:
    if value is None and supported is not None:
        return "TP" if _as_bool(supported) else "FP"
    if isinstance(value, bool):
        return "TP" if value else "FP"

    cleaned = str(value or "").strip().casefold().replace("_", " ").replace("-", " ")
    if cleaned in {"tp", "true positive", "supported", "grounded", "entailed"}:
        return "TP"
    if cleaned in {
        "fp",
        "false positive",
        "unsupported",
        "not supported",
        "hallucinated",
        "hallucination",
        "fabricated",
    }:
        return "FP"
    if cleaned in {"fn", "false negative", "missing", "omitted", "incomplete"}:
        return "FN"
    return "UNKNOWN"


def _normalise_search_section(data: Mapping[str, Any]) -> dict[str, Any]:
    section = _mapping(
        _first_value(data, ("search", "search_evaluation", "retrieval", "retrieval_evaluation"))
    )
    if not section:
        section = dict(data)

    raw_documents = _first_value(
        section,
        (
            "documents",
            "document_grades",
            "items",
            "results",
            "retrieved_documents",
            "source_relevance",
        ),
    )
    if raw_documents is None:
        raw_documents = _first_value(section, ("relevance_grades", "grades", "scores"))

    if isinstance(raw_documents, Mapping):
        document_items: list[Any] = [
            {"source_id": key, "relevance_grade": value}
            for key, value in raw_documents.items()
        ]
    elif isinstance(raw_documents, Sequence) and not isinstance(
        raw_documents, (str, bytes, bytearray)
    ):
        document_items = list(raw_documents)
    else:
        document_items = []

    documents: list[dict[str, Any]] = []
    for default_rank, item in enumerate(document_items, start=1):
        if isinstance(item, Mapping):
            item_mapping = dict(item)
            grade_value = _first_value(
                item_mapping,
                ("relevance_grade", "grade", "relevance", "score", "rating", "relevant"),
            )
            rank_number = _number(item_mapping.get("rank"))
            rank = int(rank_number) if rank_number and rank_number > 0 else default_rank
            source_id = _first_value(
                item_mapping,
                ("source_id", "document_id", "id", "source", "name"),
            )
            reason = _first_value(item_mapping, ("reason", "explanation", "rationale"))
        else:
            grade_value = item
            rank = default_rank
            source_id = None
            reason = None

        documents.append(
            {
                "rank": rank,
                "source_id": str(source_id) if source_id is not None else f"S{default_rank}",
                "relevance_grade": _normalise_grade(grade_value),
                "reason": str(reason) if reason is not None else "",
            }
        )

    documents.sort(key=lambda document: document["rank"])
    rewritten_query = _first_value(
        section,
        ("rewritten_query", "suggested_query", "query_rewrite", "improved_query"),
    )
    reason = _first_value(section, ("reason", "explanation", "summary"))
    answerable = _first_value(
        section,
        ("answerable_from_sources", "answerable", "has_answer", "context_sufficient"),
    )
    return {
        "documents": documents,
        "rewritten_query": str(rewritten_query or ""),
        "reason": str(reason or ""),
        "answerable_from_sources": (
            _as_bool(answerable) if answerable is not None else None
        ),
    }


def _normalise_source_support(answer: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_sources = _first_value(
        answer,
        ("source_support", "sources", "source_judgments", "grounding_sources"),
    )
    if isinstance(raw_sources, Mapping):
        items: list[Any] = [
            {"source_id": source_id, "supports_answer": supported}
            for source_id, supported in raw_sources.items()
        ]
    elif isinstance(raw_sources, Sequence) and not isinstance(
        raw_sources, (str, bytes, bytearray)
    ):
        items = list(raw_sources)
    else:
        items = []

    normalised: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, Mapping):
            item_mapping = dict(item)
            source_id = _first_value(
                item_mapping,
                ("source_id", "document_id", "id", "source", "name"),
            )
            support = _first_value(
                item_mapping,
                ("supports_answer", "useful", "supported", "is_useful", "is_supporting", "score"),
            )
            reason = _first_value(item_mapping, ("reason", "explanation", "rationale"))
        else:
            source_id = None
            support = item
            reason = None

        normalised.append(
            {
                "source_id": str(source_id) if source_id is not None else f"S{index}",
                "supports_answer": _as_bool(support),
                "reason": str(reason or ""),
            }
        )
    return normalised


def _claim_from_item(item: Any, default_label: str | None = None) -> dict[str, str]:
    if isinstance(item, Mapping):
        item_mapping = dict(item)
        claim = _first_value(item_mapping, ("claim", "text", "statement", "content"))
        label_value = _first_value(
            item_mapping,
            ("classification", "label", "verdict", "type", "status"),
        )
        supported = _first_value(
            item_mapping,
            ("supported", "is_supported", "grounded", "entailed"),
        )
        reason = _first_value(item_mapping, ("reason", "explanation", "rationale"))
    else:
        claim = item
        label_value = default_label
        supported = None
        reason = None

    label = _normalise_claim_label(label_value, supported)
    if label == "UNKNOWN" and default_label:
        label = default_label
    return {
        "claim": str(claim or ""),
        "label": label,
        "reason": str(reason or ""),
    }


def _normalise_claims(answer: Mapping[str, Any]) -> tuple[list[dict[str, str]], dict[str, int]]:
    raw_claims = _first_value(answer, ("claims", "claim_judgments", "claim_classifications"))
    if isinstance(raw_claims, Mapping):
        claim_items: list[Any] = [
            {"claim": claim, "classification": label}
            for claim, label in raw_claims.items()
        ]
    elif isinstance(raw_claims, Sequence) and not isinstance(
        raw_claims, (str, bytes, bytearray)
    ):
        claim_items = list(raw_claims)
    else:
        claim_items = []

    claims = [_claim_from_item(item) for item in claim_items]

    # Some judges return separate lists rather than a unified claims array.
    if not claims:
        for keys, label in (
            (("supported_claims", "true_positives"), "TP"),
            (("unsupported_claims", "false_positives", "hallucinated_claims"), "FP"),
            (("missing_claims", "missing_key_facts", "false_negatives"), "FN"),
        ):
            values = _first_value(answer, keys)
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
                claims.extend(_claim_from_item(item, label) for item in values)
    elif not any(claim["label"] == "FN" for claim in claims):
        missing = _first_value(
            answer,
            ("missing_claims", "missing_key_facts", "false_negatives"),
        )
        if isinstance(missing, Sequence) and not isinstance(missing, (str, bytes, bytearray)):
            claims.extend(_claim_from_item(item, "FN") for item in missing)

    counts = {
        label.casefold(): sum(claim["label"] == label for claim in claims)
        for label in ("TP", "FP", "FN")
    }

    canonical_counts = _mapping(answer.get("claim_counts"))
    for label in ("tp", "fp", "fn"):
        if counts[label] == 0:
            canonical_count = _count(canonical_counts.get(label))
            if canonical_count is not None:
                counts[label] = canonical_count

    # Count-only judge responses remain useful.  Explicit counts fill labels
    # that are absent, but never override evidence already represented by the
    # normalized claim list.
    count_keys = {
        "tp": ("tp_count", "true_positive_count", "tp", "true_positives"),
        "fp": ("fp_count", "false_positive_count", "fp", "false_positives"),
        "fn": ("fn_count", "false_negative_count", "fn", "false_negatives"),
    }
    for label, keys in count_keys.items():
        if counts[label] == 0:
            explicit = _count(_first_value(answer, keys))
            if explicit is not None:
                counts[label] = explicit

    return claims, counts


def _reported_percent(answer: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    value = _number(_first_value(answer, keys))
    if value is None:
        return None
    # Accept either 0--1 proportions or 0--100 percentages.
    if 0.0 <= value <= 1.0:
        value *= 100.0
    return min(100.0, max(0.0, value))


def _normalise_answer_section(data: Mapping[str, Any]) -> dict[str, Any]:
    answer = _mapping(
        _first_value(data, ("answer", "answer_evaluation", "generation", "generation_evaluation"))
    )
    if not answer:
        answer = dict(data)

    source_support = _normalise_source_support(answer)
    claims, claim_counts = _normalise_claims(answer)

    relevance_value = _first_value(
        answer,
        ("relevancy", "relevance", "relevant", "is_relevant", "helpful", "direct"),
    )
    noncommittal = _first_value(
        answer,
        ("noncommittal", "non_committal", "is_noncommittal", "evasive"),
    )
    reason = _first_value(answer, ("reason", "explanation", "summary"))
    answerable = _first_value(
        answer,
        ("answerable_from_sources", "answerable", "has_answer", "context_sufficient"),
    )

    canonical_reported = _mapping(answer.get("reported_metrics"))

    def reported(keys: Sequence[str]) -> float | None:
        direct = _reported_percent(answer, keys)
        return direct if direct is not None else _reported_percent(canonical_reported, keys)

    return {
        "source_support": source_support,
        "claims": claims,
        "claim_counts": claim_counts,
        "relevancy": binary_relevancy(relevance_value, noncommittal=noncommittal),
        "answerable_from_sources": (
            _as_bool(answerable) if answerable is not None else None
        ),
        "reason": str(reason or ""),
        "reported_metrics": {
            "grounding_percent": reported(
                ("grounding_percent", "grounding_percentage", "grounding_score"),
            ),
            "tp_percent": reported(
                ("tp_percent", "supported_claims_percent", "precision_percent"),
            ),
            "fp_percent": reported(
                ("fp_percent", "hallucination_percent", "hallucinated_claims_percent"),
            ),
            "claim_completeness_percent": reported(
                ("claim_completeness_percent", "completeness_percent", "claim_recall_percent"),
            ),
        },
    }


def normalize_combined_judgment(raw: Any) -> dict[str, Any]:
    """Normalize flexible combined-judge JSON into one stable schema.

    Canonical output:

    ``search.documents``
        Ranked items with ``source_id``, ``relevance_grade`` (0--3), and reason.
    ``answer.source_support``
        Source-level boolean support judgments.
    ``answer.claims``
        Claim-level labels ``TP``, ``FP``, ``FN``, or ``UNKNOWN``.
    ``answer.relevancy``
        Binary 0/1 directness and helpfulness judgment.
    """

    data = extract_json_object(raw)
    return {
        "search": _normalise_search_section(data),
        "answer": _normalise_answer_section(data),
    }


def search_grades_from_judgment(raw: Any) -> list[float]:
    """Extract ranked 0--3 document grades from a raw combined judgment."""

    normalised = normalize_combined_judgment(raw)
    return [
        document["relevance_grade"]
        for document in normalised["search"]["documents"]
    ]


def _percentage(numerator: int, denominator: int) -> float:
    return (100.0 * numerator / denominator) if denominator else 0.0


def derive_answer_metrics(raw: Any) -> dict[str, Any]:
    """Derive grounding and claim metrics from raw judge labels.

    ``TP`` and ``FP`` percentages use claims made by the generated answer as
    their denominator.  Claim completeness is ``TP / (TP + FN)``.  If a judge
    supplies only aggregate percentages, those values are used as a fallback;
    source/claim labels always take precedence.
    """

    answer = normalize_combined_judgment(raw)["answer"]
    sources = answer["source_support"]
    supporting_sources = sum(source["supports_answer"] for source in sources)

    counts = answer["claim_counts"]
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    reported = answer["reported_metrics"]

    grounding = (
        _percentage(supporting_sources, len(sources))
        if sources
        else (reported["grounding_percent"] or 0.0)
    )
    asserted_claims = tp + fp
    tp_percent = (
        _percentage(tp, asserted_claims)
        if asserted_claims
        else (reported["tp_percent"] or 0.0)
    )
    fp_percent = (
        _percentage(fp, asserted_claims)
        if asserted_claims
        else (reported["fp_percent"] or 0.0)
    )
    expected_claims = tp + fn
    completeness = (
        _percentage(tp, expected_claims)
        if expected_claims
        else (reported["claim_completeness_percent"] or 0.0)
    )

    return {
        "grounding_percent": grounding,
        "tp_percent": tp_percent,
        "fp_percent": fp_percent,
        # Descriptive aliases make UI/config code readable while retaining the
        # TP/FP terminology used in the lesson.
        "supported_claims_percent": tp_percent,
        "hallucinated_claims_percent": fp_percent,
        "claim_completeness_percent": completeness,
        "relevancy": answer["relevancy"],
        "source_count": len(sources),
        "supporting_source_count": supporting_sources,
        "claim_counts": {"tp": tp, "fp": fp, "fn": fn},
    }


def evaluate_combined_judgment(
    raw: Any,
    k: int | None = None,
    *,
    total_relevant: int | None = None,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    graded_ndcg: bool = False,
) -> dict[str, Any]:
    """Normalize one judge response and derive its search and answer metrics."""

    normalised = normalize_combined_judgment(raw)
    grades = [
        document["relevance_grade"]
        for document in normalised["search"]["documents"]
    ]
    return {
        "judgment": normalised,
        "search": evaluate_search_metrics(
            grades,
            k,
            total_relevant=total_relevant,
            threshold=threshold,
            graded_ndcg=graded_ndcg,
        ),
        "answer": derive_answer_metrics(raw),
    }
