# RAG Evaluation Learning Lab

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/workflow-LangGraph-green.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

An inspectable learning project for adding evaluation to an existing fitness and nutrition RAG pipeline. It retrieves from local PDF embeddings, generates a source-constrained answer, evaluates retrieval and answer quality, and exposes the scores and bounded retry path to the learner.

This repository demonstrates evaluation techniques; it does not claim production accuracy or guarantee that every answer is free of hallucinations.

## What this project teaches

- How retrieval quality and answer quality fail independently.
- How an LLM-as-a-judge can provide live quality signals when no labels exist.
- How a labeled benchmark enables reproducible MRR, MAP, recall, ROUGE, and fuzzy scores.
- How LangGraph can route low-quality results through one bounded retry.
- Why evaluation scores, thresholds, costs, and limitations should be visible to users.

The knowledge base remains the fitness and sports-nutrition collection from the original RAG project. FAISS stores embeddings produced locally with `all-MiniLM-L6-v2`; `gpt-4.1-mini` generates answers and performs live judging.

## Evaluated RAG flow

```text
User question
    |
    v
Retrieve ranked, unique source pages
    |
    v
Search evaluator (LLM judge -> Python metrics)
    | pass
    |------------------------------.
    | fail and retry available     |
    v                              |
Rewrite query -> retrieve once ----'
    |
    v
Generate a source-constrained answer
    |
    v
Answer evaluator (LLM judge -> Python metrics)
    | pass
    |------------------------------.
    | fail and retry available     |
    v                              |
Stricter prompt -> generate once --'
    |
    v
Answer + sources + scores + retry counts
```

`config.yaml` limits the graph to one search retry and one answer retry. These bounds keep the control flow understandable and prevent an evaluator from creating an infinite or unexpectedly expensive loop.

## Two evaluation modes

The application deliberately separates live estimates from labeled benchmark metrics.

| Mode | Evidence available | Metrics shown |
| --- | --- | --- |
| Live chat | The question, retrieved text, and generated answer | Judged source grades, precision at k, reciprocal rank, NDCG at k, grounding/source usefulness, supported-claim percentage, hallucinated-claim percentage, and answer relevancy |
| Labeled benchmark | A reference answer, expected source-page IDs, and an answerability label | MRR, MAP, precision, recall, F1, NDCG, ROUGE-1, ROUGE-2, ROUGE-L, fuzzy similarity, plus the live judge checks |

Recall and MAP require a known set of relevant items. ROUGE and fuzzy matching require a reference answer. Those values therefore belong to the labeled benchmark and should be displayed as `N/A`, not zero, for an arbitrary live question.

The live judge grades observations; deterministic Python functions perform the metric arithmetic. This makes the formulas independently testable and keeps model-generated math out of the scoring path.

### Teaching thresholds from the supplied evaluation lesson

These are the lesson's exact “good value” defaults, not fitness-domain guarantees. Calibrate them against a larger, human-reviewed dataset before using them as real quality gates.

| Pillar | Metric | Range | Teaching target |
| --- | --- | ---: | ---: |
| Search | MRR / reciprocal rank | 0–1 | > 0.8 |
| Search | MAP | 0–1 | > 0.7 |
| Search | Precision | 0–1 | > 0.8 |
| Search | Recall | 0–1 | > 0.7 |
| Search | F1 | 0–1 | > 0.7 |
| Search | NDCG | 0–1 | > 0.8 |
| Answer | ROUGE-1 | 0–1 | > 0.5 |
| Answer | ROUGE-2 | 0–1 | > 0.3 |
| Answer | ROUGE-L | 0–1 | > 0.4 |
| Answer | Fuzzy similarity | 0–100 | > 70 |
| Answer judge | Grounding / useful sources | 0–100% | > 80% |
| Answer judge | Supported claims (TP) | 0–100% | > 90% |
| Answer judge | Hallucinated claims (FP) | 0–100% | < 10% |
| Answer judge | Relevancy | 0 or 1 | = 1 |

In live search evaluation, relevance is estimated only among retrieved results. That supports precision, first-relevant rank, and ranking-quality signals, but it cannot prove that every relevant item in the full database was found.

## Cost and latency

With evaluation enabled, a successful no-retry chat turn makes three LLM calls:

1. Search judge.
2. Answer generation.
3. Answer judge.

That is two additional LLM calls compared with answer generation alone. A search retry adds another search-judge call. An answer retry adds another generation and answer-judge call. The graph allows each retry at most once.

Actual cost depends on source length, answer length, the configured judge model, and current provider pricing. Inspect your OpenAI usage rather than relying on a fixed per-query estimate.

## Quick start

### Prerequisites

- Python 3.10 or newer.
- An OpenAI API key.
- Internet access on first use to download the local embedding model.
- Text-extractable PDFs. Scanned documents require OCR before ingestion.

### Install

```bash
git clone https://github.com/NisargKadam/RAG_Evaluation.git
cd RAG_Evaluation

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell activation:

```powershell
.\venv\Scripts\Activate.ps1
```

Use a fresh virtual environment because this project pins LangChain 0.3.x packages.

### Configure

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder:

```dotenv
OPENAI_API_KEY=your-openai-api-key-here
```

The real `.env` is ignored by Git. Never commit an API key.

### Ingest the PDFs

Place PDF files in `data/pdfs/`, then build the FAISS index:

```bash
python app/ingest.py recursive
```

Re-run ingestion after changing documents, embeddings, or chunking. Ingestion now adds stable `document_id` and `chunk_id` metadata used by the benchmark.

Available learning strategies are:

```bash
python app/ingest.py token_based
python app/ingest.py semantic
python app/ingest.py agentic
python app/ingest.py recursive
```

Agentic chunking uses the OpenAI API; the other listed strategies split locally.

### Open the evaluated chat UI

```bash
streamlit run streamlit_app.py
```

The Streamlit response includes the answer, source pages, quality status, live search and answer metrics, and whether either bounded retry ran. Detailed score explanations are kept collapsible so the answer remains easy to read.

For the terminal chat:

```bash
python -m app.main
```

## Run the labeled benchmark

The benchmark dataset stores 12 manually checked questions, reference answers, expected source-page identifiers, and whether each question is answerable from the knowledge base. Ten are answerable and two are deliberate hard negatives.

```bash
python run_evaluation.py
```

Use benchmark results to compare retrieval or prompting changes. Keep the dataset fixed during a comparison, and review labels manually when a score is surprising.

See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for the ground-truth page convention, benchmark inventory, formulas, and evaluator caveats.

## Run the tests

```bash
python -m unittest discover -s tests -v
```

The tests should remain deterministic: metric arithmetic and graph routing can be checked with fixed observations or mocked judges instead of spending API credits.

## What the UI scores mean

- **Precision at k:** fraction of retrieved pages the live judge marked relevant.
- **Reciprocal rank:** how early the first judged-relevant page appeared.
- **NDCG at k:** whether the strongest evidence was ranked near the top.
- **Grounding:** percentage of retrieved sources judged useful for supporting the answer.
- **Supported claims:** percentage of answer claims supported by at least one source.
- **Hallucinated claims:** percentage of answer claims not supported by the supplied sources.
- **Relevancy:** whether the answer is direct when evidence exists, or correctly abstains when it does not.

Source-page IDs are de-duplicated before live evaluation. Several chunks from one page should not masquerade as several independent relevant documents.

## Configuration

The main controls live in `config.yaml`:

```yaml
retrieval:
  top_k: 3
  candidate_multiplier: 3

evaluation:
  enabled: true
  judge_model: "gpt-4.1-mini"
  max_search_retries: 1
  max_answer_retries: 1
  benchmark_file: "data/evaluation_questions.json"
  thresholds:
    precision_at_k: 0.8
    reciprocal_rank: 0.8
    ndcg_at_k: 0.8
    grounding_percent: 80
    supported_claims_percent: 90
    hallucinated_claims_percent: 10
    relevancy: 1
```

FAISS returns a distance where lower is better. A raw distance is not a confidence percentage, so the evaluated retriever does not present it as one. Tune retrieval settings with the labeled benchmark instead of copying an arbitrary similarity cutoff.

## Project structure

```text
RAG_Evaluation/
├── app/
│   ├── agent.py                 # Evaluated LangGraph workflow
│   ├── graph.py                 # Shared graph state
│   ├── judge.py                 # Structured live LLM-judge prompts
│   ├── evaluation.py            # Deterministic metric functions
│   ├── benchmarks.py            # Dataset loading and page-ID matching
│   ├── ingest.py                # PDF ingestion and stable IDs
│   ├── chunking_strategies.py   # Four learning strategies
│   └── main.py                  # Terminal chat
├── data/
│   ├── pdfs/                    # Fitness and nutrition sources
│   └── evaluation_questions.json
├── tests/                       # Unit and workflow tests
├── run_evaluation.py            # Labeled benchmark runner
├── streamlit_app.py             # Evaluated chat UI
├── EVALUATION_GUIDE.md           # Metrics and benchmark notes
├── config.yaml
├── requirements.txt
└── .env.example
```

`vectorstore/` is generated locally and ignored by Git.

## Limitations

- The included benchmark is intentionally small and educational; its scores do not establish production accuracy.
- LLM judges can be inconsistent or wrong. Temperature zero reduces variation but does not remove it.
- A judge and generator from the same model family may share blind spots.
- Live MRR-like and NDCG signals use judge-assigned relevance within the retrieved set; full recall and MAP need labels.
- ROUGE and fuzzy matching reward surface overlap and may under-score valid paraphrases or miss a changed number.
- A high evaluation score is evidence, not proof, that an answer is correct.
- Evaluation adds latency and API cost, especially when retries run.
- The ingestion pipeline extracts text but does not OCR scanned pages.
- Thresholds copied from the lesson are teaching defaults and are not calibrated to this fitness corpus.

## Troubleshooting

### `OPENAI_API_KEY is not set`

Copy `.env.example` to `.env`, add the key, and restart Streamlit. Do not add quotes or commit the real file.

### The FAISS index is missing, stale, or has no stable IDs

Rebuild it:

```bash
python app/ingest.py recursive
```

Both `vectorstore/index.faiss` and `vectorstore/index.pkl` are required.

### The embedding model repeatedly tries to contact Hugging Face

The first run must download `all-MiniLM-L6-v2`. After it is cached, offline environments can set:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### The judge is unavailable

Check the API key, provider connectivity, rate limits, and model access. The application may still return an answer, but unavailable evaluations must be treated as missing—not passing scores.

### A scanned PDF contributes no chunks

Run OCR on the PDF, replace it with a text-extractable version, and ingest again.

### LangChain dependency conflicts

Create a brand-new virtual environment and install only `requirements.txt`. Reusing a system, Conda base, or unrelated environment commonly leaves incompatible LangChain 1.x packages installed beside this project's pinned 0.3.x packages.

### `No module named app`

Run commands from the repository root. Use module mode for the terminal chat:

```bash
python -m app.main
```

## License and attribution

The code is available under the [MIT License](LICENSE). The evaluation thresholds and teaching structure are adapted from the supplied “RAG Evaluation: Search Evaluator & Answer Evaluator” lesson.

Built by **Nisarg Kadam** for hands-on AI learning.
