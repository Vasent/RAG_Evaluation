"""Streamlit interface for the RAG Evaluation Learning Lab."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import streamlit as st
from dotenv import load_dotenv

from app.benchmarks import load_benchmarks


load_dotenv()

st.set_page_config(
    page_title="RAG Evaluation Learning Lab",
    page_icon="🧪",
    layout="wide",
)

VECTORSTORE_FILES = (
    Path("./vectorstore/index.faiss"),
    Path("./vectorstore/index.pkl"),
)
BENCHMARK_FILE = Path("data/evaluation_questions.json")


def check_api_key() -> bool:
    """Return whether an OpenAI key is available without exposing it."""
    return bool(os.getenv("OPENAI_API_KEY"))


@st.cache_data(show_spinner=False)
def load_benchmark_records() -> List[Dict[str, Any]]:
    """Load and validate the labeled learning benchmark."""
    return load_benchmarks(BENCHMARK_FILE)


@st.cache_resource(
    show_spinner=(
        "Loading the knowledge base. The first embedding-model download can take a minute…"
    )
)
def load_agent():
    """Create the cached agent, ingesting when either FAISS file is absent."""
    if not all(path.exists() for path in VECTORSTORE_FILES):
        from app.ingest import DocumentIngestion

        DocumentIngestion(strategy="recursive").run()

    missing = [str(path) for path in VECTORSTORE_FILES if not path.exists()]
    if missing:
        raise RuntimeError(
            "Knowledge-base ingestion did not create: " + ", ".join(missing)
        )

    from app.agent import RAGAgent

    return RAGAgent()


def source_title(source_meta: Dict[str, Any]) -> str:
    """Return a readable source title while preserving page metadata."""
    raw = str(source_meta.get("source", "Unknown"))
    return Path(raw).stem.replace("-", " ").replace("_", " ").title()


def decimal_value(value: Any, digits: int = 3) -> str:
    """Format a numeric metric or return N/A for an undefined value."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def percent_value(value: Any) -> str:
    """Format a 0–100 metric without confusing an absent value with zero."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def distance_value(value: Any) -> str:
    """Describe the raw FAISS result correctly as a lower-is-better distance."""
    if value is None:
        return "distance unavailable"
    try:
        return f"FAISS distance {float(value):.3f} (lower is better)"
    except (TypeError, ValueError):
        return "distance unavailable"


def render_metric_row(metrics: Iterable[tuple[str, str]]) -> None:
    """Render a compact row of Streamlit metric cards."""
    entries = list(metrics)
    if not entries:
        return
    columns = st.columns(len(entries))
    for column, (label, value) in zip(columns, entries):
        with column:
            st.metric(label, value)


def render_overall_status(evaluation: Dict[str, Any]) -> None:
    """Show one compact quality state followed by bounded retry counts."""
    overall = evaluation.get("overall", {})
    status = overall.get("status", "unknown")

    if status == "passed":
        st.success("Evaluation passed")
    elif status == "needs_review":
        st.warning("Needs review — one or more teaching thresholds were not met")
    elif status == "evaluator_error":
        st.error(
            "Evaluator error — the answer is shown, but missing scores are not treated as a pass"
        )
    elif status == "disabled":
        st.info("Evaluation disabled for this response")
    else:
        st.warning("Evaluation status unavailable")

    search_retries = int(overall.get("search_retries", 0) or 0)
    answer_retries = int(overall.get("answer_retries", 0) or 0)
    st.caption(
        f"Retries — search: {search_retries}/1 · answer: {answer_retries}/1"
    )


def render_sources(sources: List[Dict[str, Any]]) -> None:
    """Show only the source pages approved for answer generation."""
    if not sources:
        st.caption("Sources: no page was approved for the final answer.")
        return

    with st.expander(f"Sources used for the answer ({len(sources)})", expanded=False):
        for source in sources:
            source_id = source.get("source_id", "?")
            page = source.get("page", "N/A")
            st.markdown(
                f"- **[{source_id}] {source_title(source)}** — page {page} · "
                f"{distance_value(source.get('distance'))}"
            )
            if source.get("preview"):
                st.caption(str(source["preview"]))


def render_search_details(search: Dict[str, Any]) -> None:
    """Render per-query retrieval metrics and judged page evidence."""
    st.markdown("#### Search evaluation")
    mode = search.get("mode")
    if mode == "benchmark_labels":
        st.caption(
            "Labeled benchmark: search relevance comes from trusted source-page labels."
        )
    elif mode == "llm_judge_estimate":
        st.caption(
            "Live values are LLM estimates for the retrieved pages, not ground truth."
        )
    elif mode == "disabled":
        st.info("Search evaluation was disabled.")
        return
    else:
        st.caption("Search metrics are unavailable or incomplete.")

    if search.get("status") == "error":
        st.caption("Search evaluator unavailable for this response.")
    elif search.get("summary"):
        st.caption(str(search["summary"]))

    metrics = search.get("metrics", {})
    render_metric_row(
        (
            ("Precision@k", decimal_value(metrics.get("precision_at_k"))),
            ("RR", decimal_value(metrics.get("reciprocal_rank"))),
            ("AP@k", decimal_value(metrics.get("average_precision_at_k"))),
            ("NDCG@k", decimal_value(metrics.get("ndcg_at_k"))),
        )
    )
    render_metric_row(
        (
            ("Recall@k", decimal_value(metrics.get("recall_at_k"))),
            ("F1", decimal_value(metrics.get("f1"))),
        )
    )
    st.caption(
        "RR is reciprocal rank for this one question; MRR is its mean across a benchmark."
    )

    items = search.get("items") or []
    if items:
        st.markdown("##### Judged retrieved pages")
        for item in items:
            source_id = item.get("source_id", "?")
            page = item.get("page", "N/A")
            grade_text = decimal_value(item.get("grade"), digits=1)
            used = "approved" if item.get("used_for_answer") else "not used"
            st.markdown(
                f"- **[{source_id}] {source_title(item)}** — page {page} · "
                f"grade {grade_text}/3 · {used} · "
                f"{distance_value(item.get('distance'))}"
            )
            if item.get("reason"):
                st.caption(f"Reason: {item['reason']}")
            if item.get("preview"):
                st.caption(str(item["preview"]))


def render_answer_details(
    answer_evaluation: Dict[str, Any],
    sources: List[Dict[str, Any]],
) -> None:
    """Render live answer-judge metrics and its source/claim reasoning."""
    st.markdown("#### Answer evaluation")
    mode = answer_evaluation.get("mode")
    if mode == "disabled":
        st.info("Answer evaluation was disabled.")
        return
    if mode == "abstention_check":
        st.caption("Abstention was checked against answer availability.")
    else:
        st.caption(
            "Grounding, claim support, hallucination, and relevancy are LLM-judge estimates."
        )

    if answer_evaluation.get("status") == "error":
        st.caption("Answer evaluator unavailable for this response.")
    elif answer_evaluation.get("summary"):
        st.caption(str(answer_evaluation["summary"]))

    metrics = answer_evaluation.get("metrics", {})
    relevancy = metrics.get("relevancy")
    relevancy_text = "N/A" if relevancy is None else f"{int(relevancy)} / 1"
    render_metric_row(
        (
            ("Grounding", percent_value(metrics.get("grounding_percent"))),
            (
                "Supported claims",
                percent_value(metrics.get("supported_claims_percent")),
            ),
            (
                "Hallucinated claims",
                percent_value(metrics.get("hallucinated_claims_percent")),
            ),
            ("Relevancy", relevancy_text),
        )
    )

    source_lookup = {
        str(source.get("source_id", "?")): source for source in sources
    }
    source_support = answer_evaluation.get("source_support") or []
    if source_support:
        st.markdown("##### Source-support reasons")
        for item in source_support:
            source_id = str(item.get("source_id", "?"))
            source = source_lookup.get(source_id, {})
            name = source_title(source) if source else source_id
            page = source.get("page", "N/A")
            verdict = (
                "supports answer"
                if item.get("supports_answer")
                else "does not support answer"
            )
            st.markdown(f"- **[{source_id}] {name}** — page {page} · {verdict}")
            if item.get("reason"):
                st.caption(f"Reason: {item['reason']}")

    claims = answer_evaluation.get("claims") or []
    if claims:
        st.markdown("##### Claim reasons")
        for item in claims:
            label = str(item.get("label", "UNKNOWN"))
            claim = str(item.get("claim", "(claim text unavailable)"))
            st.markdown(f"- **{label}** — {claim}")
            if item.get("reason"):
                st.caption(f"Reason: {item['reason']}")


def rouge_f1(reference_metrics: Dict[str, Any], metric_name: str) -> Any:
    """Read a nested ROUGE F1 value safely."""
    metric = reference_metrics.get(metric_name)
    return metric.get("f1") if isinstance(metric, dict) else None


def render_reference_details(reference_metrics: Dict[str, Any]) -> None:
    """Render metrics that are valid only when a trusted reference exists."""
    if not reference_metrics:
        return

    st.markdown("#### Reference-answer metrics")
    st.caption(
        "Available for the selected labeled benchmark; these compare the answer with its trusted reference."
    )
    render_metric_row(
        (
            ("ROUGE-1 F1", decimal_value(rouge_f1(reference_metrics, "rouge_1"))),
            ("ROUGE-2 F1", decimal_value(rouge_f1(reference_metrics, "rouge_2"))),
            ("ROUGE-L F1", decimal_value(rouge_f1(reference_metrics, "rouge_l"))),
        )
    )
    render_metric_row(
        (
            ("Fuzzy", percent_value(reference_metrics.get("fuzzy_ratio"))),
            (
                "Fuzzy partial",
                percent_value(reference_metrics.get("fuzzy_partial_ratio")),
            ),
            (
                "Semantic similarity",
                decimal_value(reference_metrics.get("semantic_similarity")),
            ),
        )
    )
    if reference_metrics.get("semantic_similarity_error"):
        st.caption("Semantic similarity unavailable for this response.")


def attempt_verdict(attempt: Dict[str, Any]) -> str:
    """Return a compact, safe label for one evaluation attempt."""
    if attempt.get("status") == "error":
        return "evaluator error"
    passed = attempt.get("passed")
    if passed is True:
        return "passed"
    if passed is False:
        return "needs review"
    return "not scored"


def render_attempt_trace(
    search: Dict[str, Any],
    answer_evaluation: Dict[str, Any],
) -> None:
    """Explain retries through compact, metric-focused attempt summaries."""
    search_attempts = search.get("attempts") or []
    answer_attempts = answer_evaluation.get("attempts") or []
    if len(search_attempts) <= 1 and len(answer_attempts) <= 1:
        return

    st.markdown("#### Retry attempt trace")
    st.caption("Each stage can retry once; the final attempt supplies the displayed answer.")

    if len(search_attempts) > 1:
        for number, attempt in enumerate(search_attempts, start=1):
            metrics = attempt.get("metrics", {})
            st.markdown(
                f"- **Search attempt {number} — {attempt_verdict(attempt)}** · "
                f"Precision@k {decimal_value(metrics.get('precision_at_k'))} · "
                f"RR {decimal_value(metrics.get('reciprocal_rank'))} · "
                f"NDCG@k {decimal_value(metrics.get('ndcg_at_k'))}"
            )

    if len(answer_attempts) > 1:
        for number, attempt in enumerate(answer_attempts, start=1):
            metrics = attempt.get("metrics", {})
            relevancy = metrics.get("relevancy")
            relevancy_text = (
                "N/A" if relevancy is None else decimal_value(relevancy, digits=0)
            )
            st.markdown(
                f"- **Answer attempt {number} — {attempt_verdict(attempt)}** · "
                f"Grounding {percent_value(metrics.get('grounding_percent'))} · "
                f"Supported {percent_value(metrics.get('supported_claims_percent'))} · "
                f"Hallucinated {percent_value(metrics.get('hallucinated_claims_percent'))} · "
                f"Relevancy {relevancy_text}"
            )


def render_evaluation_details(
    result: Dict[str, Any],
    evaluation: Dict[str, Any],
) -> None:
    """Render detailed scoring without placing it before the answer."""
    with st.expander("Evaluation details", expanded=False):
        search = evaluation.get("search", {})
        answer_evaluation = evaluation.get("answer", {})

        if result.get("retrieval_query") and result.get("retrieval_query") != result.get(
            "question"
        ):
            st.caption(f"Final retrieval query: {result['retrieval_query']}")

        render_attempt_trace(search, answer_evaluation)
        render_search_details(search)
        st.divider()
        render_answer_details(answer_evaluation, result.get("sources", []))
        render_reference_details(answer_evaluation.get("reference_metrics", {}))


def render_assistant_result(result: Dict[str, Any]) -> None:
    """Keep the answer first, then show status, sources, and evaluation."""
    st.markdown(str(result.get("answer", "No answer was returned.")))
    evaluation = result.get("evaluation")
    if not isinstance(evaluation, dict):
        st.warning("Evaluation details were not returned.")
        render_sources(result.get("sources", []))
        return

    render_overall_status(evaluation)
    render_sources(result.get("sources", []))
    render_evaluation_details(result, evaluation)


def render_history_message(message: Dict[str, Any]) -> None:
    """Render a saved chat message, including its original evaluation."""
    if message.get("role") == "assistant" and isinstance(message.get("result"), dict):
        render_assistant_result(message["result"])
        return
    if message.get("error"):
        st.error(str(message["error"]))
        return

    st.markdown(str(message.get("content", "")))
    if message.get("benchmark_id"):
        st.caption(f"Labeled benchmark: {message['benchmark_id']}")


try:
    benchmark_records = load_benchmark_records()
    benchmark_error: Optional[str] = None
except (OSError, ValueError, TypeError) as exc:
    benchmark_records = []
    benchmark_error = str(exc)

benchmark_by_id = {
    str(record["id"]): record for record in benchmark_records
}

with st.sidebar:
    st.title("🧪 RAG Evaluation Learning Lab")
    st.markdown(
        "Compare live LLM-judge estimates with reproducible, labeled benchmark metrics."
    )
    st.divider()

    evaluation_enabled = st.toggle(
        "Enable evaluation",
        value=True,
        help="Turn off to run the original retrieve-and-generate path.",
    )
    st.caption(
        "Normal no-retry path: evaluation adds 2 LLM calls "
        "(one search judge and one answer judge)."
    )

    st.divider()
    st.subheader("Labeled benchmark")
    if benchmark_error:
        st.error(f"Could not load benchmark data: {benchmark_error}")

    benchmark_ids = [""] + list(benchmark_by_id)

    def benchmark_label(benchmark_id: str) -> str:
        if not benchmark_id:
            return "Choose a benchmark question"
        record = benchmark_by_id[benchmark_id]
        availability = "answerable" if record.get("answerable") else "unanswerable"
        return f"{benchmark_id} · {availability} · {record['question']}"

    selected_benchmark_id = st.selectbox(
        "Benchmark question",
        options=benchmark_ids,
        format_func=benchmark_label,
        disabled=not benchmark_records,
    )
    run_benchmark = st.button(
        "Run benchmark",
        use_container_width=True,
        disabled=not selected_benchmark_id or not evaluation_enabled,
    )
    if selected_benchmark_id and not evaluation_enabled:
        st.caption("Enable evaluation to run the labeled benchmark.")

    st.divider()
    st.subheader("Indexed text sources")
    st.markdown(
        "- Exercise & Nutrition\n"
        "- Sports Nutrition Fundamentals\n"
        "- High Calorie & High Protein Education Sheet"
    )
    st.caption("The image-only sample PDF requires OCR before it can be indexed.")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.title("RAG Evaluation Learning Lab")
st.caption(
    "Fitness RAG with visible retrieval checks, answer checks, and bounded retries"
)

if not check_api_key():
    st.error(
        "**OPENAI_API_KEY is not set.**  \n"
        "Copy .env.example to .env, add the key, and restart the app."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for saved_message in st.session_state.messages:
    with st.chat_message(saved_message.get("role", "assistant")):
        render_history_message(saved_message)

try:
    agent = load_agent()
except Exception as exc:
    st.error(
        "Could not load the RAG agent. Check the API key, embedding-model cache, "
        f"and both FAISS index files. Error type: {type(exc).__name__}."
    )
    st.stop()

request_question: Optional[str] = None
request_benchmark: Optional[Dict[str, Any]] = None

if run_benchmark and selected_benchmark_id:
    request_benchmark = benchmark_by_id[selected_benchmark_id]
    request_question = str(request_benchmark["question"])

chat_prompt = st.chat_input(
    "Ask about fitness, nutrition, or sports performance…"
)
if request_question is None and chat_prompt:
    request_question = chat_prompt.strip()

if request_question:
    user_message: Dict[str, Any] = {
        "role": "user",
        "content": request_question,
    }
    if request_benchmark:
        user_message["benchmark_id"] = request_benchmark["id"]
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        render_history_message(user_message)

    with st.chat_message("assistant"):
        spinner_text = (
            "Running the labeled benchmark…"
            if request_benchmark
            else "Retrieving, answering, and evaluating…"
        )
        try:
            query_kwargs: Dict[str, Any] = {"evaluate": evaluation_enabled}
            if request_benchmark:
                query_kwargs.update(
                    {
                        "reference_answer": request_benchmark["reference_answer"],
                        "relevant_documents": request_benchmark[
                            "relevant_documents"
                        ],
                        "answerable": request_benchmark["answerable"],
                    }
                )

            with st.spinner(spinner_text):
                query_result = agent.query(request_question, **query_kwargs)

            render_assistant_result(query_result)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "result": query_result,
                }
            )
        except Exception as exc:
            error_message = (
                "The request could not be completed. Check provider connectivity, "
                "model access, and the local index. "
                f"Error type: {type(exc).__name__}."
            )
            st.error(error_message)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "error": error_message,
                }
            )
