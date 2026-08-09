
"""Shared state for the evaluated RAG LangGraph workflow."""

from typing import Any, Dict, List, Optional, TypedDict

from langchain.schema import Document


class GraphState(TypedDict, total=False):
    """
    State carried through retrieval, generation, evaluation, and bounded retries.

    ``total=False`` keeps the graph easy to inspect while allowing optional
    benchmark fields (reference answers and gold document labels) to be passed
    only when they are available.
    """

    question: str
    retrieval_query: str
    context: List[Document]
    approved_context: List[Document]
    answer: str
    sources: List[Dict[str, Any]]
    evaluation_enabled: bool
    reference_answer: Optional[str]
    relevant_documents: List[Dict[str, Any]]
    answerable: Optional[bool]
    search_evaluation: Dict[str, Any]
    answer_evaluation: Dict[str, Any]
    search_history: List[Dict[str, Any]]
    answer_history: List[Dict[str, Any]]
    search_retries: int
    answer_retries: int
    strict_generation: bool
    needs_search_retry: bool
    needs_answer_retry: bool
