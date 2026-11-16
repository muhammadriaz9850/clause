# Clause

A customer-support assistant for Clause Shop, built with RAG (Retrieval-Augmented Generation). It runs locally with Ollama by default.

Ask a question and Clause looks it up in the shop handbook. It answers only from what it finds and returns the sources it used. If the handbook doesn't cover the question, it says "I don't know."

## How it works

```
shop_handbook.pdf → chunk → dense embed (Ollama) + sparse BM25 (FastEmbed) → Qdrant
                                                                               │
question ──→ hybrid search (dense + BM25, top-k, score threshold) ────────────┘
                 │
                 └──→ LLM (handbook context only) ──→ answer + sources
```

- **Hybrid retrieval**: semantic vectors and BM25 keyword matching are combined. Exact terms like prices, day counts and product names are matched more reliably this way.
- **Grounded answers**: the system prompt makes the model answer only from the retrieved handbook text.
- **Evals**: a golden test set measures retrieval hit rate, answer correctness, refusals and leaks of staff-only information.

## Project layout

| Folder        | What it holds                                                  |
|---------------|----------------------------------------------------------------|
| `api/`        | FastAPI RAG backend (LangChain + Ollama + Qdrant), evals, agent prototype |
| `web/`        | Frontend (coming soon)                                         |
| `notebooks/`  | Step-by-step learning notebooks (local only, not tracked)      |
| `playground/` | Standalone RAG experiments (local only, not tracked)           |

## Quick start

```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text

cd api
uv sync
cp .env.example .env
uv run python -m app.services.ingestion   # build the vector store
uv run fastapi dev app/main.py            # start the API
```

Then open http://127.0.0.1:8000/docs and try `POST /ask`.

To measure answer quality, run `uv run python -m evals.run` (stop the API first).

See [`api/README.md`](api/README.md) for endpoints, configuration, evals and the CLI.
