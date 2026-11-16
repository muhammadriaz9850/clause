# Clause API

A FastAPI backend that answers customer questions from the handbook using RAG (Retrieval-Augmented Generation).

It loads the handbook PDF and splits it into chunks. Each chunk is indexed in Qdrant twice: as a dense embedding and as a sparse BM25 vector. Questions are answered from the chunks that hybrid search ranks as most relevant. By default everything runs locally with Ollama.

## Tech stack

- **FastAPI**: HTTP API
- **LangChain**: model, embedding and agent setup (`init_chat_model`, `init_embeddings`, `create_agent`)
- **Ollama**: local chat model (`qwen3.5:4b`) and dense embeddings (`nomic-embed-text`)
- **FastEmbed**: sparse BM25 embeddings (`Qdrant/bm25`) for keyword matching
- **Qdrant**: vector store with hybrid dense + sparse search (embedded on disk by default, or a server)
- **pydantic-settings**: configuration from `.env`

## Project layout

```
api/
├── app/
│   ├── main.py            # FastAPI app and routes
│   ├── cli.py             # Ask a question from the terminal
│   ├── core/
│   │   └── config.py      # Settings, embeddings and Qdrant client
│   ├── schemas/
│   └── services/
│       ├── ingestion.py   # Load → chunk → embed (dense + BM25) → store
│       ├── chat.py        # RAGService: hybrid retrieve → prompt → answer
│       └── agent.py       # Prototype LangChain agent (experimental)
├── evals/
│   ├── golden.jsonl       # Golden test questions
│   ├── run.py             # Eval harness
│   └── results/           # Timestamped JSON reports
├── files/
│   └── shop_handbook.pdf  # Source document
├── vector_stores/         # Local vector store (created by ingestion, not tracked)
├── .env.example
└── pyproject.toml
```

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally

Pull the models:

```bash
ollama pull qwen3.5:4b
ollama pull nomic-embed-text
```

The BM25 model (`Qdrant/bm25`) is downloaded automatically by FastEmbed the first time it is used.

## Setup

Run all commands from the `api/` folder.

```bash
cd api
uv sync
cp .env.example .env
```

## 1. Ingest the handbook

This builds the Qdrant collection from `files/shop_handbook.pdf`, with a `dense` vector and a `sparse` (BM25) vector for each chunk.

```bash
uv run python -m app.services.ingestion
```

Run it once, and again whenever the PDF or the chunking and embedding settings change. It deletes and rebuilds the collection each time. Before deleting anything, it checks that the embedding model's output size matches `RAG_EMBED_DIM`.

## 2. Run the API

```bash
uv run fastapi dev app/main.py
```

The API runs at http://127.0.0.1:8000. Interactive docs are at http://127.0.0.1:8000/docs.

### Endpoints

| Method | Path   | Description                          |
|--------|--------|--------------------------------------|
| GET    | `/`    | Health check: `{"status": "ok"}`     |
| POST   | `/ask` | Ask a question about the handbook    |

**Request**

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?"}'
```

`question` is required. Whitespace is stripped, and it must be 1–2000 characters long.

**Response**

```json
{
  "answer": "…",
  "sources": [
    { "source": "files/shop_handbook.pdf", "page": 0, "score": 0.5 }
  ]
}
```

`score` is the hybrid search score after dense and BM25 results are fused. If no chunk scores at least `RAG_SCORE_THRESHOLD`, the API skips the LLM call. It returns `"I don't know."` with an empty `sources` list.

## Ask from the terminal

```bash
uv run python -m app.cli
```

This asks one question and prints the answer.

> **Note:** embedded Qdrant (`RAG_QDRANT_PATH`) can only be opened by one process at a time. Stop the API before running the CLI, ingestion or evals, or point `RAG_QDRANT_URL` at a Qdrant server.

## Evals

The eval harness runs every question in `evals/golden.jsonl` through `RAGService` and grades the answers:

```bash
uv run python -m evals.run
```

There are three kinds of test case:

| Type           | Passes when                                                     |
|----------------|-----------------------------------------------------------------|
| `answerable`   | The answer contains every `must_include` term (`a\|b` means either) |
| `out_of_scope` | The model refuses, for example with "I don't know"              |
| `restricted`   | The answer contains none of the `must_not_include` staff-only terms |

It prints a pass/fail line for each case. Each failing answerable case gets a blame note. The note says whether the retriever missed the fact or the generator got it wrong even though the fact was in the chunks. The summary metrics are:

- **Retrieval hit rate**: a source came from one of the expected pages
- **Fact in chunks**: the required terms appear in the retrieved text
- **Answer correctness**: the answer contains the required terms
- **Refusal rate**: out-of-scope questions were refused
- **Leak rate**: restricted questions leaked staff-only information

Each run is saved to `evals/results/<timestamp>.json` together with the config it used (models, chunking, `top_k`, threshold and prompt), so you can compare runs side by side.

## Agent (experimental)

`app/services/agent.py` is an early prototype of a LangChain agent built with `create_agent`. It uses a local Ollama model and has no tools yet. It isn't connected to the API. To try it from the terminal:

```bash
uv run python -m app.services.agent
```

## Configuration

All settings are read from `api/.env`, and each one needs the `RAG_` prefix. See [`.env.example`](.env.example) for the full list. The defaults below come from `app/core/config.py`.

| Variable               | Default                   | Purpose                                      |
|------------------------|---------------------------|----------------------------------------------|
| `RAG_APP_NAME`         | `Clause`                  | App name                                     |
| `RAG_DATA_PATH`        | `files/shop_handbook.pdf` | Document to ingest                           |
| `RAG_CHAT_PROVIDER`    | `ollama`                  | Chat model provider                          |
| `RAG_CHAT_MODEL`       | `gemma3:1b`               | Chat model name                              |
| `RAG_CHAT_TEMPERATURE` | `0.1`                     | Sampling temperature                         |
| `RAG_CHAT_API_KEY`     | *(empty)*                 | API key for hosted providers                 |
| `RAG_EMBED_PROVIDER`   | `ollama`                  | Dense embedding provider                     |
| `RAG_EMBED_MODEL`      | `nomic-embed-text:latest` | Dense embedding model                        |
| `RAG_EMBED_DIM`        | `768`                     | Must match the embedding model's output size |
| `RAG_CHUNK_SIZE`       | `600`                     | Characters per chunk                         |
| `RAG_CHUNK_OVERLAP`    | `100`                     | Overlap between chunks                       |
| `RAG_TOP_K`            | `3`                       | Chunks retrieved per question                |
| `RAG_SCORE_THRESHOLD`  | `0.5`                     | Minimum hybrid score for a chunk to be used  |
| `RAG_COLLECTION_NAME`  | `shop_handbook`           | Qdrant collection                            |
| `RAG_QDRANT_PATH`      | `./vector_stores`         | Local Qdrant storage                         |
| `RAG_QDRANT_URL`       | *(unset)*                 | Qdrant server URL (overrides the path)       |
| `RAG_SYSTEM_PROMPT`    | *(see config.py)*         | Grounding prompt for the chat model          |

To use a hosted provider, install its LangChain package and set the provider, model and API key as shown in `.env.example`. For Google Gemini (`langchain-google-genai` is already installed), set `RAG_CHAT_PROVIDER=google_genai`, a Gemini model name and `RAG_CHAT_API_KEY`.

If you change the embedding model, update `RAG_EMBED_DIM` and run ingestion again. Run the evals after any retrieval or prompt change to check that quality hasn't dropped.
