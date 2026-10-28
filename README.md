# Simple RAG with Qdrant & NVIDIA Nemotron

A single-file, production-grade Minimalist Retrieval-Augmented Generation (RAG) system in Python. It pairs a **Qdrant** vector database with fast local embeddings (**FastEmbed**) and NVIDIA's **Nemotron-3** LLM.

---

## Architecture & Data Flow

```
1. Document Ingestion:
   [PDF / DOCX] ──> Text Extraction ──> Sliding Window Chunking (1000 chars, 200 overlap)
                          │
                          ▼
   FastEmbed (BAAI/bge-small-en-v1.5) ──> Qdrant Vector DB (Cosine Metric)

2. Query & Retrieval:
   User Question ──> FastEmbed Vector ──> Qdrant Similarity Search (Top-K)
                          │
                          ▼
   Context + Prompt ──> NVIDIA Nemotron LLM ──> Answer
```

---

## Tech Stack & Design Choices

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Vector DB** | [Qdrant](https://qdrant.tech/) | Production-ready vector search engine with native REST/gRPC API and built-in Web Dashboard. |
| **Embeddings** | [FastEmbed](https://github.com/qdrant/fastembed) (`BAAI/bge-small-en-v1.5`) | CPU-optimized ONNX runtime embeddings; no heavy PyTorch overhead; compatible with Python 3.12–3.14. |
| **LLM** | NVIDIA `nemotron-3-ultra-550b-a55b` | High-capability reasoning model accessed through NVIDIA's OpenAI-compatible API. |
| **Package Manager** | [uv](https://docs.astral.sh/uv/) | Ultra-fast Python package and project manager. |

---

## Prerequisites

- **Python**: 3.12+ (tested up to Python 3.14)
- **[uv](https://docs.astral.sh/uv/)**: Recommended package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **[Docker](https://docs.docker.com/get-docker/)**: For running local Qdrant
- **NVIDIA API Key**: Obtain one from [build.nvidia.com](https://build.nvidia.com)

---

## 1. Setup & Installation

### Clone and Configure Environment

```bash
git clone https://github.com/harshsen007/Simple_RAG.git
cd Simple_RAG
```

Create a `.env` file in the project root:

```bash
NVIDIA_API_KEY="your-nvidia-api-key-here"
# Optional (defaults to http://localhost:6333)
QDRANT_URL="http://localhost:6333"
```

### Install Dependencies

Using `uv`:

```bash
uv sync
```

*(Or using standard pip: `pip install -e .`)*

---

## 2. Running Qdrant Vector Database in Docker

Qdrant runs as a standalone service in Docker.

### Option A: Standard Run with Persistent Storage (Recommended)

Stores your vectors and collections in `./qdrant_storage` so data persists across container restarts:

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant
```

### Option B: Ephemeral Run (Testing / In-Memory)

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Managing the Qdrant Container

- **Check if running**:
  ```bash
  docker ps -f name=qdrant
  ```
- **View container logs**:
  ```bash
  docker logs -f qdrant
  ```
- **Stop Qdrant**:
  ```bash
  docker stop qdrant
  ```
- **Start Qdrant**:
  ```bash
  docker start qdrant
  ```
- **Remove container**:
  ```bash
  docker rm -f qdrant
  ```

---

## 3. How to Check Qdrant Localhost

Once the Docker container is active:

### 1. Web Dashboard (Visual UI)
Open your browser and navigate to:
```
http://localhost:6333/dashboard
```
Here you can:
- Explore the `strategy` collection.
- Inspect individual point vectors, chunk text, and metadata payloads.
- View collection telemetry and execute test vector queries.

### 2. REST API / Terminal Check
Verify the health and collections via `curl`:

```bash
# Server status and version
curl http://localhost:6333/

# Inspect collection info and point counts
curl http://localhost:6333/collections/strategy
```

---

## 4. Usage

### Step 1: Add Documents
Place your `.pdf` or `.docx` documents inside the [`Plan_docs/`](file:///home/lap-77/projects/Simple_rag/Plan_docs) directory.

### Step 2: Ingest & Index Documents
Run the ingestion command to parse documents, chunk the text, compute embeddings, and load them into Qdrant:

```bash
uv run rag.py ingest
```

*Example output:*
```
Constructing_Strategies_for_Programming.pdf: stored 46 chunks
```

### Step 3: Ask Questions
Query your documents from the command line:

```bash
uv run rag.py ask "insertion sort"
```

*Example output:*
```
Retrieved chunks:

- [chunk: 17 | score: 0.7606] herwise = x : insert a xs Figure 1: Model solution for insertion sort completing the definition is to realise that we have to distinguish the empty list from the list with at least one argument...

- [chunk: 23 | score: 0.7504] igure 2 formulates at a very high level what is essential about insertion sort: we start with the empty list ( foldr’s second argument), and the helper-functioninsert is used for each element...

- [chunk: 27 | score: 0.7333] tions (such as foldr and para). Programming tasks that are expressed with the same higher-order function essentially belong to the same problem class...

Answer:

Based on the provided context, insertion sort is implemented in Haskell with two main approaches:

1. Explicit recursion with helper function (Figure 1):
   - Base case: isort [] = []
   - Recursive case: isort (x:xs) = insert x (isort xs)
...
```

---

## 5. Configuration Settings

All configurations are centralized near the top of [`rag.py`](file:///home/lap-77/projects/Simple_rag/rag.py):

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `DOCS_DIR` | `Plan_docs/` | Directory where source PDF and DOCX files reside. |
| `QDRANT_URL` | `http://localhost:6333` | URL of the Qdrant instance (reads from `QDRANT_URL` env if set). |
| `COLLECTION_NAME` | `"strategy"` | Qdrant collection name. |
| `EMBEDDING_MODEL_NAME` | `"BAAI/bge-small-en-v1.5"` | Local FastEmbed model (384 dimensions, Cosine metric). |
| `MODEL` | `"nvidia/nemotron-3-ultra-550b-a55b"` | LLM model identifier. |
| `BASE_URL` | `https://integrate.api.nvidia.com/v1` | NVIDIA OpenAI-compatible API base URL. |
| `TOP_K` | `5` | Number of most relevant chunks to retrieve. |
| `MAX_CHUNK_CHARS` | `1000` | Size of each chunk window in characters. |
| `CHUNK_OVERLAP` | `200` | Overlap size between adjacent chunk windows. |

---

## Project Structure

```
Simple_rag/
├── .env                  # Environment variables (NVIDIA_API_KEY, QDRANT_URL)
├── .gitignore            # Git ignore list
├── Plan_docs/            # Source documents directory (.pdf / .docx)
├── pyproject.toml        # Dependencies and project definition
├── rag.py                # Complete RAG pipeline script
├── README.md             # Documentation and usage guide
└── uv.lock               # Deterministic dependency lockfile
```
