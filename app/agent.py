"""Evaluated Retrieval-Augmented Generation agent.

The LangGraph workflow keeps live LLM judgments separate from deterministic
benchmark labels.  Search and answer retries are deliberately bounded so the
learning project stays inspectable and cost-aware.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.benchmarks import metadata_document_id, relevant_document_ids
from app.evaluation import (
    cosine_similarity,
    derive_answer_metrics,
    evaluate_search_metrics,
    fuzzy_partial_ratio,
    fuzzy_ratio,
    normalize_combined_judgment,
    rouge_scores,
)
from app.graph import GraphState
from app.judge import LLMJudge


load_dotenv()


ABSTENTION = "I don't have information about that in the provided documents."


class RAGAgent:
    """Retrieve, answer, evaluate, and retry through one compiled graph."""

    def __init__(
        self,
        config_path: str = "config.yaml",
        *,
        embeddings: Any = None,
        vectorstore: Any = None,
        llm: Any = None,
        judge_llm: Any = None,
    ):
        with open(config_path, "r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)

        self.evaluation_config = self.config.get("evaluation", {})
        self.thresholds = self.evaluation_config.get("thresholds", {})
        self.relevance_grade = float(self.evaluation_config.get("relevance_grade", 2))

        # Dependency injection keeps graph routing testable without downloads or
        # paid model calls.
        if embeddings is None and vectorstore is None:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.config["embeddings"]["model_name"]
            )
        self.embeddings = embeddings

        self.vectorstore = vectorstore or FAISS.load_local(
            self.config["vectordb"]["persist_directory"],
            self.embeddings,
            allow_dangerous_deserialization=True,  # Loads this project's own index.
        )

        self.llm = llm or ChatOpenAI(
            model=self.config["llm"]["model"],
            temperature=self.config["llm"]["temperature"],
        )
        judge_model = judge_llm or ChatOpenAI(
            model=self.evaluation_config.get("judge_model", self.config["llm"]["model"]),
            temperature=0,
        )
        self.judge = LLMJudge(
            judge_model,
            max_source_chars=int(self.evaluation_config.get("max_source_chars", 1600)),
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a careful fitness and nutrition assistant.

Answer the user's question using ONLY the supplied context.

Rules:
1. If the context supports an answer, respond clearly and concisely.
2. If it does not, respond exactly: "I don't have information about that in the provided documents."
3. Do not add facts from memory or general knowledge.
4. Cite supporting source IDs such as [S1] after the claims they support.
5. Never turn a topic mention into a recommendation that the source does not give.
{strict_rules}

Context:
{context}
""",
                ),
                ("human", "{question}"),
            ]
        )

        self.graph = self.build_graph()

    # ------------------------------------------------------------------
    # Retrieval and search evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _unique_page_results(
        results: Iterable[Any],
        limit: int,
    ) -> List[tuple[Document, Optional[float]]]:
        """Keep the best-ranked chunk from each page to avoid metric inflation."""
        unique: List[tuple[Document, Optional[float]]] = []
        seen: set[str] = set()

        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                document, raw_distance = result
                try:
                    distance = float(raw_distance)
                except (TypeError, ValueError):
                    distance = None
            else:
                document, distance = result, None

            key = metadata_document_id(document.metadata)
            if key in seen:
                continue
            seen.add(key)
            unique.append((document, distance))
            if len(unique) >= limit:
                break
        return unique

    @staticmethod
    def _source_record(
        document: Document,
        rank: int,
        distance: Optional[float],
    ) -> Dict[str, Any]:
        metadata = document.metadata
        return {
            "source_id": f"S{rank}",
            "document_id": metadata_document_id(metadata),
            "chunk_id": metadata.get("chunk_id"),
            "source": Path(str(metadata.get("source", "Unknown"))).name,
            "page": metadata.get("page", "N/A"),
            "rank": rank,
            "distance": distance,
            "preview": " ".join(document.page_content.split())[:280],
        }

    def retrieve_node(self, state: GraphState) -> GraphState:
        """Retrieve ranked candidates and collapse duplicate source pages."""
        retrieval_query = state.get("retrieval_query") or state["question"]
        retry_count = int(state.get("search_retries", 0))
        base_k = int(self.config["retrieval"]["top_k"])
        requested_k = base_k * (retry_count + 1)
        multiplier = max(1, int(self.config["retrieval"].get("candidate_multiplier", 3)))
        candidate_k = requested_k * multiplier

        if hasattr(self.vectorstore, "similarity_search_with_score"):
            raw_results = self.vectorstore.similarity_search_with_score(
                retrieval_query,
                k=candidate_k,
            )
        else:  # Useful for small fakes in unit tests.
            raw_results = self.vectorstore.similarity_search(
                retrieval_query,
                k=candidate_k,
            )

        ranked = self._unique_page_results(raw_results, requested_k)
        context = [document for document, _ in ranked]
        sources = [
            self._source_record(document, rank, distance)
            for rank, (document, distance) in enumerate(ranked, start=1)
        ]

        state["retrieval_query"] = retrieval_query
        state["context"] = context
        state["sources"] = sources
        return state

    def _search_passed(self, metrics: Dict[str, Any]) -> bool:
        return (
            metrics.get("precision_at_k", 0.0)
            >= float(self.thresholds.get("precision_at_k", 0.8))
            and metrics.get("reciprocal_rank", 0.0)
            >= float(self.thresholds.get("reciprocal_rank", 0.8))
            and metrics.get("ndcg_at_k", 0.0)
            >= float(self.thresholds.get("ndcg_at_k", 0.8))
        )

    def evaluate_search_node(self, state: GraphState) -> GraphState:
        """Use gold page labels when supplied; otherwise use an LLM judge."""
        context = state.get("context", [])
        sources = [dict(source) for source in state.get("sources", [])]
        retry_count = int(state.get("search_retries", 0))
        state["needs_search_retry"] = False

        if not state.get("evaluation_enabled", True):
            for source in sources:
                source["used_for_answer"] = True
            state["approved_context"] = context
            state["sources"] = sources
            state["search_evaluation"] = {
                "status": "disabled",
                "mode": "disabled",
                "passed": None,
                "metrics": {},
                "items": [],
                "retries": retry_count,
                "attempts": [],
            }
            return state

        relevant_documents = state.get("relevant_documents")
        raw_judgment: Dict[str, Any] = {}
        total_relevant: Optional[int] = None

        if relevant_documents is not None:
            mode = "benchmark_labels"
            gold_ids = relevant_document_ids(relevant_documents)
            total_relevant = len(gold_ids)
            grades = [
                3.0 if metadata_document_id(document.metadata) in gold_ids else 0.0
                for document in context
            ]
            items = [
                {
                    "source_id": f"S{rank}",
                    "rank": rank,
                    "grade": grade,
                    "reason": (
                        "Matches a labeled relevant source page."
                        if grade >= self.relevance_grade
                        else "Not in the labeled relevant source-page set."
                    ),
                }
                for rank, grade in enumerate(grades, start=1)
            ]
            rewritten_query = state["question"]
            answerable_from_sources = any(grade >= self.relevance_grade for grade in grades)
        else:
            mode = "llm_judge_estimate"
            raw_judgment = self.judge.evaluate_search(state["question"], context)
            if raw_judgment.get("error"):
                for source in sources:
                    source["used_for_answer"] = True
                state["approved_context"] = context
                state["sources"] = sources
                evaluation = {
                    "status": "error",
                    "mode": mode,
                    "passed": None,
                    "metrics": {},
                    "items": [],
                    "summary": raw_judgment["error"],
                    "retries": retry_count,
                }
                history_item = dict(evaluation)
                history = list(state.get("search_history", [])) + [history_item]
                evaluation["attempts"] = history
                state["search_history"] = history
                state["search_evaluation"] = evaluation
                return state

            normalized = normalize_combined_judgment(raw_judgment)["search"]
            judgment_by_id = {
                item["source_id"]: item for item in normalized["documents"]
            }
            items = []
            grades = []
            for rank in range(1, len(context) + 1):
                source_id = f"S{rank}"
                item = judgment_by_id.get(source_id, {})
                grade = float(item.get("relevance_grade", 0.0))
                grades.append(grade)
                items.append(
                    {
                        "source_id": source_id,
                        "rank": rank,
                        "grade": grade,
                        "reason": item.get("reason", "No judge reason returned."),
                    }
                )
            rewritten_query = normalized.get("rewritten_query") or state["question"]
            judged_answerable = normalized.get("answerable_from_sources")
            answerable_from_sources = (
                bool(judged_answerable)
                if judged_answerable is not None
                else any(grade >= self.relevance_grade for grade in grades)
            )

        metrics = evaluate_search_metrics(
            grades,
            total_relevant=total_relevant,
            threshold=self.relevance_grade,
            graded_ndcg=True,
        )
        # Classical AP/NDCG are undefined for a labeled query with no relevant
        # documents. Correct rejection is evaluated through abstention instead.
        if mode == "benchmark_labels" and total_relevant == 0:
            metrics["average_precision_at_k"] = None
            metrics["ndcg_at_k"] = None

        relevant_positions = {
            item["rank"]
            for item in items
            if item["grade"] >= self.relevance_grade
        }
        approved_context = [
            document
            for rank, document in enumerate(context, start=1)
            if rank in relevant_positions
        ]
        for source, item in zip(sources, items):
            source.update(
                {
                    "relevance_grade": item["grade"],
                    "relevance_reason": item["reason"],
                    "used_for_answer": item["rank"] in relevant_positions,
                }
            )

        is_labeled_unanswerable = mode == "benchmark_labels" and total_relevant == 0
        is_judged_unanswerable = (
            mode == "llm_judge_estimate"
            and not answerable_from_sources
            and not relevant_positions
        )
        correct_rejection = is_labeled_unanswerable or is_judged_unanswerable
        passed = True if correct_rejection else self._search_passed(metrics)
        evaluation = {
            "status": "complete",
            "mode": mode,
            "passed": passed,
            "metrics": metrics,
            "items": [
                {**source, **item}
                for source, item in zip(sources, items)
            ],
            "answerable_from_sources": answerable_from_sources,
            "summary": (
                "Correctly found no supporting document."
                if correct_rejection
                else "Search met the configured teaching thresholds."
                if passed
                else "Search is below one or more teaching thresholds."
            ),
            "retries": retry_count,
        }
        history_item = {key: value for key, value in evaluation.items() if key != "attempts"}
        history = list(state.get("search_history", [])) + [history_item]
        evaluation["attempts"] = history

        approved_sources = [source for source in sources if source["used_for_answer"]]
        for answer_rank, source in enumerate(approved_sources, start=1):
            source["retrieval_source_id"] = source["source_id"]
            source["source_id"] = f"S{answer_rank}"

        state["approved_context"] = approved_context
        state["sources"] = approved_sources
        state["search_history"] = history
        state["search_evaluation"] = evaluation

        max_retries = int(self.evaluation_config.get("max_search_retries", 1))
        if not passed and retry_count < max_retries:
            state["search_retries"] = retry_count + 1
            state["retrieval_query"] = str(rewritten_query).strip() or state["question"]
            state["needs_search_retry"] = True
        return state

    @staticmethod
    def route_after_search(state: GraphState) -> str:
        if state.get("needs_search_retry"):
            return "retry"
        if state.get("approved_context"):
            return "generate"
        return "abstain"

    # ------------------------------------------------------------------
    # Generation and answer evaluation
    # ------------------------------------------------------------------

    def generate_node(self, state: GraphState) -> GraphState:
        """Generate from evaluator-approved context only."""
        context = state.get("approved_context", state.get("context", []))
        if not context:
            state["answer"] = ABSTENTION
            return state

        context_text = "\n\n".join(
            f"[S{rank}] {Path(str(document.metadata.get('source', 'Unknown'))).name}, "
            f"page {document.metadata.get('page', 'N/A')}\n{document.page_content}"
            for rank, document in enumerate(context, start=1)
        )
        strict_rules = ""
        if state.get("strict_generation"):
            strict_rules = (
                "6. This is a retry: split the draft into claims, remove every claim "
                "that lacks direct support, and prefer a partial grounded answer over guessing."
            )

        response = (self.prompt | self.llm).invoke(
            {
                "context": context_text,
                "question": state["question"],
                "strict_rules": strict_rules,
            }
        )
        state["answer"] = response.content if hasattr(response, "content") else str(response)
        return state

    def _reference_metrics(self, answer: str, reference: str) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            **rouge_scores(reference, answer),
            "fuzzy_ratio": fuzzy_ratio(reference, answer),
            "fuzzy_partial_ratio": fuzzy_partial_ratio(reference, answer),
            "semantic_similarity": None,
        }
        if self.embeddings is not None:
            try:
                reference_vector, answer_vector = self.embeddings.embed_documents(
                    [reference, answer]
                )
                metrics["semantic_similarity"] = cosine_similarity(
                    reference_vector,
                    answer_vector,
                )
            except Exception as exc:
                metrics["semantic_similarity_error"] = str(exc)
        return metrics

    def _answer_passed(self, metrics: Dict[str, Any]) -> bool:
        return (
            metrics.get("grounding_percent", 0.0)
            >= float(self.thresholds.get("grounding_percent", 80))
            and metrics.get("supported_claims_percent", 0.0)
            >= float(self.thresholds.get("supported_claims_percent", 90))
            and metrics.get("hallucinated_claims_percent", 100.0)
            <= float(self.thresholds.get("hallucinated_claims_percent", 10))
            and metrics.get("relevancy", 0)
            >= int(self.thresholds.get("relevancy", 1))
        )

    def evaluate_answer_node(self, state: GraphState) -> GraphState:
        """Judge source support and claims, then optionally request one retry."""
        retry_count = int(state.get("answer_retries", 0))
        state["needs_answer_retry"] = False

        if not state.get("evaluation_enabled", True):
            state["answer_evaluation"] = {
                "status": "disabled",
                "mode": "disabled",
                "passed": None,
                "metrics": {},
                "reference_metrics": {},
                "retries": retry_count,
                "attempts": [],
            }
            return state

        context = state.get("approved_context", [])
        raw_judgment = self.judge.evaluate_answer(
            state["question"],
            state.get("answer", ""),
            context,
        )
        if raw_judgment.get("error"):
            reference = state.get("reference_answer")
            evaluation = {
                "status": "error",
                "mode": "llm_judge",
                "passed": None,
                "metrics": {},
                "reference_metrics": (
                    self._reference_metrics(state.get("answer", ""), reference)
                    if reference
                    else {}
                ),
                "summary": raw_judgment["error"],
                "retries": retry_count,
            }
            history_item = dict(evaluation)
            history = list(state.get("answer_history", [])) + [history_item]
            evaluation["attempts"] = history
            state["answer_history"] = history
            state["answer_evaluation"] = evaluation
            return state

        normalized = normalize_combined_judgment(raw_judgment)["answer"]
        derived = derive_answer_metrics(raw_judgment)
        support_by_id = {
            item["source_id"]: item for item in normalized["source_support"]
        }
        complete_source_support = [
            support_by_id.get(
                f"S{rank}",
                {
                    "source_id": f"S{rank}",
                    "supports_answer": False,
                    "reason": "Judge omitted this source.",
                },
            )
            for rank in range(1, len(context) + 1)
        ]
        supporting_source_count = sum(
            bool(item["supports_answer"]) for item in complete_source_support
        )
        grounding_percent = (
            100.0 * supporting_source_count / len(complete_source_support)
            if complete_source_support
            else 0.0
        )
        metrics = {
            "grounding_percent": grounding_percent,
            "supported_claims_percent": derived["tp_percent"],
            "hallucinated_claims_percent": derived["fp_percent"],
            "claim_completeness_percent": derived["claim_completeness_percent"],
            "relevancy": derived["relevancy"],
            "source_count": len(complete_source_support),
            "supporting_source_count": supporting_source_count,
            "claim_counts": derived["claim_counts"],
        }
        reference = state.get("reference_answer")
        reference_metrics = (
            self._reference_metrics(state.get("answer", ""), reference)
            if reference
            else {}
        )
        passed = self._answer_passed(metrics)
        evaluation = {
            "status": "complete",
            "mode": "llm_judge",
            "passed": passed,
            "metrics": metrics,
            "reference_metrics": reference_metrics,
            "source_support": complete_source_support,
            "claims": normalized["claims"],
            "answerable_from_sources": normalized["answerable_from_sources"],
            "summary": normalized.get("reason") or (
                "Answer met the configured teaching thresholds."
                if passed
                else "Answer is below one or more teaching thresholds."
            ),
            "retries": retry_count,
        }
        history_item = {key: value for key, value in evaluation.items() if key != "attempts"}
        history = list(state.get("answer_history", [])) + [history_item]
        evaluation["attempts"] = history
        state["answer_history"] = history
        state["answer_evaluation"] = evaluation

        max_retries = int(self.evaluation_config.get("max_answer_retries", 1))
        if not passed and retry_count < max_retries:
            state["answer_retries"] = retry_count + 1
            state["strict_generation"] = True
            state["needs_answer_retry"] = True
        return state

    @staticmethod
    def route_after_answer(state: GraphState) -> str:
        return "retry" if state.get("needs_answer_retry") else "finish"

    def abstain_node(self, state: GraphState) -> GraphState:
        """Return a safe fallback and score expected abstention separately."""
        state["answer"] = ABSTENTION
        expected_answerable = state.get("answerable")
        correct_abstention = expected_answerable is not True
        search_mode = state.get("search_evaluation", {}).get("mode", "llm_judge_estimate")
        state["answer_evaluation"] = {
            "status": "complete",
            "mode": "abstention_check",
            "passed": correct_abstention,
            "correct_abstention": correct_abstention,
            "metrics": {
                "grounding_percent": None,
                "supported_claims_percent": None,
                "hallucinated_claims_percent": 0.0,
                "claim_completeness_percent": None,
                "relevancy": 1 if correct_abstention else 0,
            },
            "reference_metrics": (
                self._reference_metrics(state["answer"], state["reference_answer"])
                if state.get("reference_answer")
                else {}
            ),
            "summary": (
                "Correct abstention: no supporting source was available."
                if correct_abstention
                else "Retrieval missed a labeled answer, so the abstention is incorrect."
            ),
            "retries": int(state.get("answer_retries", 0)),
            "attempts": [],
            "search_mode": search_mode,
        }
        return state

    # ------------------------------------------------------------------
    # Graph and public API
    # ------------------------------------------------------------------

    def build_graph(self):
        """Compile the evaluated workflow once at agent startup."""
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("evaluate_search", self.evaluate_search_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("evaluate_answer", self.evaluate_answer_node)
        workflow.add_node("abstain", self.abstain_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "evaluate_search")
        workflow.add_conditional_edges(
            "evaluate_search",
            self.route_after_search,
            {"retry": "retrieve", "generate": "generate", "abstain": "abstain"},
        )
        workflow.add_edge("generate", "evaluate_answer")
        workflow.add_conditional_edges(
            "evaluate_answer",
            self.route_after_answer,
            {"retry": "generate", "finish": END},
        )
        workflow.add_edge("abstain", END)
        return workflow.compile()

    def query(
        self,
        question: str,
        *,
        evaluate: Optional[bool] = None,
        reference_answer: Optional[str] = None,
        relevant_documents: Optional[List[Dict[str, Any]]] = None,
        answerable: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Run one question and return answer, citations, and evaluation details."""
        evaluation_enabled = (
            bool(self.evaluation_config.get("enabled", True))
            if evaluate is None
            else bool(evaluate)
        )
        initial: GraphState = {
            "question": question,
            "retrieval_query": question,
            "context": [],
            "approved_context": [],
            "answer": "",
            "sources": [],
            "evaluation_enabled": evaluation_enabled,
            "search_history": [],
            "answer_history": [],
            "search_retries": 0,
            "answer_retries": 0,
            "strict_generation": False,
            "needs_search_retry": False,
            "needs_answer_retry": False,
        }
        if reference_answer is not None:
            initial["reference_answer"] = reference_answer
        if relevant_documents is not None:
            initial["relevant_documents"] = relevant_documents
        if answerable is not None:
            initial["answerable"] = answerable

        result = self.graph.invoke(initial)
        search = result.get("search_evaluation", {})
        answer = result.get("answer_evaluation", {})
        if not evaluation_enabled:
            overall_status, overall_passed = "disabled", None
        elif search.get("status") == "error" or answer.get("status") == "error":
            overall_status, overall_passed = "evaluator_error", None
        else:
            overall_passed = bool(search.get("passed")) and bool(answer.get("passed"))
            overall_status = "passed" if overall_passed else "needs_review"

        return {
            "question": question,
            "retrieval_query": result.get("retrieval_query", question),
            "answer": result.get("answer", ABSTENTION),
            "sources": result.get("sources", []),
            "evaluation": {
                "search": search,
                "answer": answer,
                "overall": {
                    "status": overall_status,
                    "passed": overall_passed,
                    "search_retries": int(result.get("search_retries", 0)),
                    "answer_retries": int(result.get("answer_retries", 0)),
                },
            },
        }
