"""Network-free behavioral tests for the evaluated LangGraph workflow."""

from __future__ import annotations

from copy import deepcopy
import json
import unittest
from typing import Any, Iterable

from langchain.schema import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.agent import ABSTENTION, RAGAgent


def make_document(
    content: str,
    source: str,
    page: int,
    chunk_id: str,
) -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "page": page, "chunk_id": chunk_id},
    )


class FakeVectorStore:
    """Return one explicitly supplied ranked result set per retrieval call."""

    def __init__(self, responses: Iterable[list[tuple[Document, float]]]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def similarity_search_with_score(
        self,
        query: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        call_index = len(self.calls)
        self.calls.append({"query": query, "k": k})
        if call_index >= len(self.responses):
            raise AssertionError("retrieval exceeded the expected bounded attempts")
        return list(self.responses[call_index])


class QueueGenerator:
    """Runnable chat-model fake that records prompts and returns queued answers."""

    def __init__(self, answers: Iterable[str]):
        self.answers = list(answers)
        self.calls: list[Any] = []
        self.runnable = RunnableLambda(self._invoke)

    def _invoke(self, prompt: Any) -> AIMessage:
        call_index = len(self.calls)
        self.calls.append(prompt)
        if call_index >= len(self.answers):
            raise AssertionError("generation exceeded the expected bounded attempts")
        return AIMessage(content=self.answers[call_index])


class QueueJudge:
    """Direct judge fake used after construction to bypass all model calls."""

    def __init__(
        self,
        search_responses: Iterable[dict[str, Any]],
        answer_responses: Iterable[dict[str, Any]],
    ):
        self.search_responses = list(search_responses)
        self.answer_responses = list(answer_responses)
        self.search_calls: list[tuple[str, list[Document]]] = []
        self.answer_calls: list[tuple[str, str, list[Document]]] = []

    def evaluate_search(
        self,
        question: str,
        documents: list[Document],
    ) -> dict[str, Any]:
        call_index = len(self.search_calls)
        self.search_calls.append((question, list(documents)))
        if call_index >= len(self.search_responses):
            raise AssertionError("search judging exceeded the expected bounded attempts")
        return deepcopy(self.search_responses[call_index])

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        documents: list[Document],
    ) -> dict[str, Any]:
        call_index = len(self.answer_calls)
        self.answer_calls.append((question, answer, list(documents)))
        if call_index >= len(self.answer_responses):
            raise AssertionError("answer judging exceeded the expected bounded attempts")
        return deepcopy(self.answer_responses[call_index])


def search_judgment(
    grades: Iterable[int],
    *,
    rewritten_query: str = "",
    answerable: bool = True,
) -> dict[str, Any]:
    return {
        "items": [
            {
                "source_id": f"S{rank}",
                "grade": grade,
                "reason": "relevant" if grade >= 2 else "not relevant",
            }
            for rank, grade in enumerate(grades, start=1)
        ],
        "rewritten_query": rewritten_query,
        "answerable_from_sources": answerable,
    }


def answer_judgment(
    source_count: int,
    labels: Iterable[str] = ("TP",),
    *,
    useful_sources: Iterable[int] | None = None,
    relevant: bool = True,
) -> dict[str, Any]:
    useful = set(useful_sources if useful_sources is not None else range(1, source_count + 1))
    return {
        "answerable_from_sources": True,
        "sources": [
            {
                "source_id": f"S{rank}",
                "useful": rank in useful,
                "reason": "supports answer" if rank in useful else "unused",
            }
            for rank in range(1, source_count + 1)
        ],
        "claims": [
            {
                "claim": f"claim {index}",
                "label": label,
                "reason": "test label",
            }
            for index, label in enumerate(labels, start=1)
        ],
        "missing_key_facts": [],
        "noncommittal": not relevant,
        "relevant": relevant,
        "summary": "test verdict",
    }


class AgentGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            make_document("Protein supports recovery.", "guide-a.pdf", 1, "a-1"),
            make_document("Carbohydrates replenish glycogen.", "guide-b.pdf", 2, "b-2"),
            make_document("Hydration supports performance.", "guide-c.pdf", 3, "c-3"),
        ]
        self.ranked = [
            (self.documents[0], 0.10),
            (self.documents[1], 0.20),
            (self.documents[2], 0.30),
        ]

    @staticmethod
    def make_agent(
        vectorstore: FakeVectorStore,
        generator: QueueGenerator,
        judge: QueueJudge,
    ) -> RAGAgent:
        # A harmless runnable prevents RAGAgent.__init__ from constructing a
        # real ChatOpenAI judge before the direct QueueJudge is installed.
        unused_judge_llm = RunnableLambda(lambda _: AIMessage(content="{}"))
        agent = RAGAgent(
            vectorstore=vectorstore,
            llm=generator.runnable,
            judge_llm=unused_judge_llm,
        )
        agent.judge = judge
        return agent

    def test_good_live_retrieval_and_answer_pass(self) -> None:
        vectorstore = FakeVectorStore([self.ranked])
        generator = QueueGenerator(["Protein supports recovery [S1]."])
        judge = QueueJudge(
            [search_judgment([3, 3, 3], answerable=True)],
            [answer_judgment(3, ["TP"])],
        )
        agent = self.make_agent(vectorstore, generator, judge)

        result = agent.query("What supports recovery?")

        self.assertEqual(result["answer"], "Protein supports recovery [S1].")
        self.assertEqual(len(result["sources"]), 3)
        self.assertEqual(result["evaluation"]["overall"]["status"], "passed")
        self.assertTrue(result["evaluation"]["search"]["passed"])
        self.assertTrue(result["evaluation"]["answer"]["passed"])
        self.assertEqual(len(vectorstore.calls), 1)
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(len(judge.search_calls), 1)
        self.assertEqual(len(judge.answer_calls), 1)

    def test_weak_search_retries_at_most_once(self) -> None:
        vectorstore = FakeVectorStore([self.ranked, self.ranked])
        generator = QueueGenerator(["A grounded partial answer [S1]."])
        judge = QueueJudge(
            [
                search_judgment([3, 0, 0], rewritten_query="recovery protein evidence"),
                search_judgment([3, 0, 0], rewritten_query="must not run again"),
            ],
            [answer_judgment(1, ["TP"])],
        )
        agent = self.make_agent(vectorstore, generator, judge)

        result = agent.query("Tell me about recovery protein")

        overall = result["evaluation"]["overall"]
        search = result["evaluation"]["search"]
        self.assertEqual(overall["search_retries"], 1)
        self.assertEqual(len(vectorstore.calls), 2)
        self.assertEqual(len(judge.search_calls), 2)
        self.assertEqual(len(search["attempts"]), 2)
        self.assertFalse(search["passed"])
        self.assertEqual(vectorstore.calls[1]["query"], "recovery protein evidence")
        self.assertEqual(len(generator.calls), 1)

    def test_false_positive_answer_retries_at_most_once(self) -> None:
        vectorstore = FakeVectorStore([self.ranked])
        generator = QueueGenerator(
            [
                "Protein supports recovery, and an unsupported claim.",
                "Retry still contains one unsupported claim [S1].",
            ]
        )
        judge = QueueJudge(
            [search_judgment([3, 3, 3])],
            [
                answer_judgment(3, ["TP", "FP"]),
                answer_judgment(3, ["TP", "FP"]),
            ],
        )
        agent = self.make_agent(vectorstore, generator, judge)

        result = agent.query("What supports recovery?")

        overall = result["evaluation"]["overall"]
        answer = result["evaluation"]["answer"]
        self.assertEqual(overall["answer_retries"], 1)
        self.assertEqual(len(generator.calls), 2)
        self.assertEqual(len(judge.answer_calls), 2)
        self.assertEqual(len(answer["attempts"]), 2)
        self.assertFalse(answer["passed"])
        self.assertGreater(answer["metrics"]["hallucinated_claims_percent"], 10)
        second_prompt = generator.calls[1].to_string()
        self.assertIn("This is a retry", second_prompt)

    def test_labeled_unanswerable_question_abstains_without_model_calls(self) -> None:
        vectorstore = FakeVectorStore([self.ranked])
        generator = QueueGenerator([])
        judge = QueueJudge([], [])
        agent = self.make_agent(vectorstore, generator, judge)

        result = agent.query(
            "What is the ACL rehabilitation protocol?",
            relevant_documents=[],
            answerable=False,
        )

        self.assertEqual(result["answer"], ABSTENTION)
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["evaluation"]["search"]["mode"], "benchmark_labels")
        self.assertTrue(result["evaluation"]["search"]["passed"])
        self.assertTrue(result["evaluation"]["answer"]["correct_abstention"])
        self.assertEqual(result["evaluation"]["overall"]["status"], "passed")
        self.assertEqual(generator.calls, [])
        self.assertEqual(judge.search_calls, [])
        self.assertEqual(judge.answer_calls, [])

    def test_retrieval_deduplicates_pages_before_generation(self) -> None:
        duplicate_page = make_document(
            "A lower-ranked duplicate chunk from the same page.",
            "guide-a.pdf",
            1,
            "a-1-duplicate",
        )
        fourth_page = make_document("Additional context.", "guide-d.pdf", 4, "d-4")
        ranked_with_duplicate = [
            (self.documents[0], 0.10),
            (duplicate_page, 0.15),
            (self.documents[1], 0.20),
            (self.documents[2], 0.30),
            (fourth_page, 0.40),
        ]
        vectorstore = FakeVectorStore([ranked_with_duplicate])
        generator = QueueGenerator(["Answer from unique pages."])
        judge = QueueJudge([], [])
        agent = self.make_agent(vectorstore, generator, judge)

        result = agent.query("A question", evaluate=False)

        document_ids = [source["document_id"] for source in result["sources"]]
        self.assertEqual(len(document_ids), 3)
        self.assertEqual(len(set(document_ids)), 3)
        self.assertEqual(
            document_ids,
            ["guide-a.pdf#page=1", "guide-b.pdf#page=2", "guide-c.pdf#page=3"],
        )
        self.assertEqual(result["sources"][0]["chunk_id"], "a-1")
        self.assertEqual(result["evaluation"]["overall"]["status"], "disabled")

    def test_evaluator_errors_do_not_discard_generated_answer(self) -> None:
        vectorstore = FakeVectorStore([self.ranked])
        generator = QueueGenerator(["The answer remains available [S1]."])
        judge = QueueJudge(
            [{"error": "search evaluator unavailable"}],
            [{"error": "answer evaluator unavailable"}],
        )
        agent = self.make_agent(vectorstore, generator, judge)

        result = agent.query("What supports recovery?")

        self.assertEqual(result["answer"], "The answer remains available [S1].")
        self.assertEqual(result["evaluation"]["search"]["status"], "error")
        self.assertEqual(result["evaluation"]["answer"]["status"], "error")
        self.assertEqual(result["evaluation"]["overall"]["status"], "evaluator_error")
        self.assertIsNone(result["evaluation"]["overall"]["passed"])
        self.assertEqual(len(generator.calls), 1)

    def test_query_result_is_json_serializable(self) -> None:
        vectorstore = FakeVectorStore([self.ranked])
        generator = QueueGenerator(["Serializable grounded answer [S1]."])
        judge = QueueJudge(
            [search_judgment([3, 3, 3])],
            [answer_judgment(3, ["TP"])],
        )
        agent = self.make_agent(vectorstore, generator, judge)

        result = agent.query("What supports recovery?")
        encoded = json.dumps(result)

        self.assertIn('"evaluation"', encoded)
        self.assertNotIn("Document(", encoded)


if __name__ == "__main__":
    unittest.main()
