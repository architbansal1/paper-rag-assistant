"""
Retrieval: given a query, return the top-k most relevant chunks from the
paper vector store.
"""
import os
import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"


class Retriever:
    def __init__(self, db_dir: str = DB_DIR, embed_model: str = EMBED_MODEL):
        self.embedder = SentenceTransformer(embed_model)
        client = chromadb.PersistentClient(path=db_dir)
        self.collection = client.get_collection("papers")

    def retrieve(self, query: str, top_k: int = 4):
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append({"text": doc, "paper": meta["paper"], "distance": dist})
        return hits

    def format_context(self, hits) -> str:
        parts = []
        for h in hits:
            parts.append(f"[Source: {h['paper']}]\n{h['text']}")
        return "\n\n".join(parts)


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What is LoRA and how does it work?"
    r = Retriever()
    hits = r.retrieve(query)
    print(f"Query: {query}\n")
    for h in hits:
        print(f"--- {h['paper']} (distance={h['distance']:.4f}) ---")
        print(h["text"][:300] + "...\n")
