"""
Pass 1: the simplest possible RAG, all in one file.

Data flow (read the file top to bottom, it follows this order):
    document -> text -> chunks -> embeddings stored in Qdrant
    question -> top matching chunks -> prompt -> LLM answer

Usage:
    python rag.py ingest
    python rag.py ask "how should I mock external calls?"
"""

import os
import sys
import uuid
from pathlib import Path

from docx import Document
from dotenv import load_dotenv
from fastembed import TextEmbedding
from openai import OpenAI
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()  # reads NVIDIA_API_KEY from the .env file in the project root

# ---- settings (all in one place so they are easy to find) ----
DOCS_DIR = Path(__file__).parent / "Plan_docs"           # where your PDF/DOCX lives
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")  # Qdrant host (Docker or local instance)
COLLECTION_NAME = "strategy"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"          # fast, high quality local embedding model
VECTOR_SIZE = 384                                        # bge-small-en-v1.5 output vector dimension
MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
BASE_URL = "https://integrate.api.nvidia.com/v1"        # NVIDIA's OpenAI-compatible API
TOP_K = 5                                               # how many chunks to retrieve
MAX_CHUNK_CHARS = 1000      # size of each chunk
CHUNK_OVERLAP = 200                                   

_embed_model = None


def get_embedding_model() -> TextEmbedding:
    global _embed_model
    if _embed_model is None:
        _embed_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    return _embed_model


def get_qdrant_client() -> QdrantClient:
    try:
        client = QdrantClient(url=QDRANT_URL)
        # verify connection
        client.get_collections()
        return client
    except Exception as exc:
        print(f"\n[Error] Could not connect to Qdrant at {QDRANT_URL}.")
        print("Make sure Qdrant is running locally, e.g. via Docker:")
        print("    docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant\n")
        raise exc


# ---------------------------------------------------------------
# Step 1: read a file and return its plain text
# ---------------------------------------------------------------
def load_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
      #  print("\n\n".join(page.extract_text() or "" for page in reader.pages))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if path.suffix.lower() == ".docx":
        doc = Document(path)
        return "\n\n".join(p.text for p in doc.paragraphs)
    raise ValueError(f"Unsupported file type: {path}")


# ---------------------------------------------------------------
# Step 2: split the text into chunks
# Simplest idea: one chunk per paragraph (blank-line separated).
# If a paragraph is huge (PDFs often have no blank lines), cut it up.
# ---------------------------------------------------------------
def chunk_text(text: str) -> list[str]:
    text = " ".join(text.split())               # collapse newlines and extra spaces
    step = MAX_CHUNK_CHARS - CHUNK_OVERLAP      # window moves 800 chars each time
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start : start + MAX_CHUNK_CHARS].strip()
        if len(chunk) >= 30:                    # skip tiny leftover tails
            chunks.append(chunk)
        if start + MAX_CHUNK_CHARS >= len(text):
            break                               # last window reached the end
    return chunks

# ---------------------------------------------------------------
# Step 3: embed the chunks and save them in Qdrant
# ---------------------------------------------------------------
def store_chunks(chunks: list[str], source: str, client: QdrantClient) -> None:
    embed_model = get_embedding_model()
    vectors = list(embed_model.embed(chunks))

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source}-{i}")),
            vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
            payload={"text": chunk, "source": source, "chunk_index": i},
        )
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def ingest() -> None:
    client = get_qdrant_client()

    # Recreate collection to start fresh so re-running doesn't duplicate
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    files = [f for f in DOCS_DIR.iterdir() if f.suffix.lower() in (".pdf", ".docx")]
    if not files:
        print(f"No PDF or DOCX files found in {DOCS_DIR}")
        return
    for f in files:
        chunks = chunk_text(load_document(f))
        store_chunks(chunks, f.name, client)
        print(f"{f.name}: stored {len(chunks)} chunks")


# ---------------------------------------------------------------
# Step 4: given a question, find the most similar chunks
# ---------------------------------------------------------------
def retrieve(question: str) -> list[dict]:
    client = get_qdrant_client()
    embed_model = get_embedding_model()
    query_vector = list(embed_model.embed([question]))[0]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector),
        limit=TOP_K,
    )
    return [
        {
            "text": hit.payload["text"],
            "score": hit.score,
            "source": hit.payload.get("source"),
            "chunk_index": hit.payload.get("chunk_index"),
        }

        for hit in results.points
    ]


# ---------------------------------------------------------------
# Step 5: put the chunks and the question into a prompt, ask the LLM
# ---------------------------------------------------------------
def answer(question: str, chunks: list[dict] | list[str] | None = None) -> str:
    if chunks is None:
        chunks = retrieve(question)

    text_chunks = [c["text"] if isinstance(c, dict) else c for c in chunks]
    context = "\n\n---\n\n".join(text_chunks)

    prompt = (
        "Answer the question using only the context below. "
        "If the context does not contain the answer, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    llm = OpenAI(base_url=BASE_URL, api_key=os.environ["NVIDIA_API_KEY"])
    response = llm.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    print("\n--- RAW RESPONSE ---")
    print(response.model_dump_json(indent=2))
    print("\n--- RAW RESPONSE END ---\n\n")
    return response.choices[0].message.content


# ---------------------------------------------------------------
# Command line entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "ingest":
        ingest()
    elif len(sys.argv) >= 3 and sys.argv[1] == "ask":
        question = " ".join(sys.argv[2:])
        chunks = retrieve(question)
        print("\nRetrieved chunks:\n")
        for c in chunks:
            text = c["text"] if isinstance(c, dict) else c
            score = c.get("score") if isinstance(c, dict) else None
            chunk_index = c.get("chunk_index") if isinstance(c, dict) else None

            meta_parts = []
            if chunk_index is not None:
                meta_parts.append(f"chunk: {chunk_index}")
            if score is not None:
                meta_parts.append(f"score: {score:.4f}")
            meta_str = f" [{' | '.join(meta_parts)}]" if meta_parts else ""

            print(f"-{meta_str} {text[:200].replace('\n', ' ')} ...\n")
        print("Answer:\n")
        print(answer(question, chunks=chunks))
    else:
        print('Usage:\n  python rag.py ingest\n  python rag.py ask "your question"')