"""
Ingest papers: extract text, chunk, embed, store in a local Chroma vector DB.

Run after download_papers.py.
"""
import os
import re
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PAPERS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "papers")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"  # small, fast, good enough for a project like this
CHUNK_SIZE = 500       # words per chunk
CHUNK_OVERLAP = 100    # words of overlap between chunks


def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def ingest():
    if not os.path.isdir(PAPERS_DIR) or not os.listdir(PAPERS_DIR):
        raise SystemExit(
            f"No PDFs found in {PAPERS_DIR}. Run download_papers.py first."
        )

    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection("papers")

    doc_id = 0
    for fname in sorted(os.listdir(PAPERS_DIR)):
        if not fname.endswith(".pdf"):
            continue
        paper_name = fname.replace(".pdf", "")
        print(f"[ingest] {paper_name}")
        raw_text = extract_text(os.path.join(PAPERS_DIR, fname))
        text = clean_text(raw_text)
        chunks = chunk_text(text)

        embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
        ids = [f"{paper_name}_{i}" for i in range(len(chunks))]
        metadatas = [{"paper": paper_name, "chunk_index": i} for i in range(len(chunks))]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        doc_id += len(chunks)
        print(f"  -> {len(chunks)} chunks added")

    print(f"Done. {doc_id} total chunks in collection 'papers' at {DB_DIR}")


if __name__ == "__main__":
    ingest()
