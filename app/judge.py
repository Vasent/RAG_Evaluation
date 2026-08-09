"""LLM-as-a-judge prompts used by the live evaluation graph.

The judge returns observations, never final percentages. Metric arithmetic is
kept in Python so learners can inspect and test it independently of the model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from langchain.schema import Document


def _source_id(document: Document, rank: int) -> str:
    """Return a short ID that is stable inside one judge request."""
    return f"S{rank}"


def _format_sources(documents: Iterable[Document], max_chars: int) -> str:
    blocks: List[str] = []
    for rank, document in enumerate(documents, start=1):
        source = Path(str(document.metadata.get("source", "Unknown"))).name
        page = document.metadata.get("page", "N/A")
        content = " ".join(document.page_content.split())[:max_chars]
        blocks.append(f"[{_source_id(document, rank)}] {source}, page {page}\n{content}")
    return "\n\n".join(blocks)


def _extract_json(text: str) -> Dict[str, Any]:
    """Parse a JSON object even when a model wraps it in Markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Judge did not return a JSON object")
        value = json.loads(cleaned[start : end + 1])

    if not isinstance(value, dict):
        raise ValueError("Judge response must be a JSON object")
    return value


class LLMJudge:
    """Run retrieval and answer checks with a low-temperature chat model."""

    def __init__(self, llm: Any, max_source_chars: int = 1600):
        self.llm = llm
        self.max_source_chars = max_source_chars

    def _invoke_json(self, prompt: str) -> Dict[str, Any]:
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            return _extract_json(content)
        except Exception as exc:  # The answer should survive an evaluator outage.
            return {"error": f"LLM judge unavailable: {exc}"}

    def evaluate_search(self, question: str, documents: List[Document]) -> Dict[str, Any]:
        """Grade every retrieved source from 0 (irrelevant) to 3 (direct)."""
        sources = _format_sources(documents, self.max_source_chars)
        prompt = f"""You evaluate retrieval quality for a RAG learning project.

Question:
{question}

Ranked retrieved sources:
{sources}

Judge each source only on whether its supplied text helps answer the question.
Use these grades consistently:
- 0: irrelevant
- 1: related topic but not useful for answering this question
- 2: partially useful or supports part of the answer
- 3: directly contains the answer or essential evidence

Return valid JSON only, with exactly one item for every source ID:
{{
  "items": [
    {{"source_id": "S1", "grade": 0, "reason": "short evidence-based reason"}}
  ],
  "rewritten_query": "a clearer retrieval query that preserves intent, or the original question",
  "answerable_from_sources": true
}}

Do not calculate metrics. Do not use outside knowledge. A source must earn at
least 2 to count as relevant. If the sources do not contain an answer, set
answerable_from_sources to false and make the rewrite concrete without adding
facts not present in the question."""
        return self._invoke_json(prompt)

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        documents: List[Document],
    ) -> Dict[str, Any]:
        """Judge source usefulness, atomic claims, omissions, and directness."""
        sources = _format_sources(documents, self.max_source_chars)
        prompt = f"""You evaluate a RAG answer using only the supplied sources.

Question:
{question}

Generated answer:
{answer}

Sources:
{sources or "(no approved sources)"}

First decide whether the question is answerable from these sources. Then:
1. Mark each source useful=true only if its text supports the generated answer.
2. Split the answer into atomic factual claims. Label each TP when supported by
   a source, or FP when unsupported. An FP is a hallucination.
3. List only important source facts needed to answer the question that the
   answer omitted (false negatives).
4. A direct answer is relevant. A clear abstention is also relevant when the
   sources genuinely do not answer the question. An evasive answer fails when
   the evidence is available.

Return valid JSON only:
{{
  "answerable_from_sources": true,
  "sources": [
    {{"source_id": "S1", "useful": true, "reason": "short reason"}}
  ],
  "claims": [
    {{"claim": "one atomic claim", "label": "TP", "source_ids": ["S1"], "reason": "short reason"}}
  ],
  "missing_key_facts": ["important omitted fact"],
  "noncommittal": false,
  "relevant": true,
  "summary": "one-sentence verdict"
}}

Do not calculate percentages. Do not reward citations unless the cited source
actually supports the claim. Do not use outside knowledge."""
        return self._invoke_json(prompt)
