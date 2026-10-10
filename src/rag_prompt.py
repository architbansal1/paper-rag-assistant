"""
Shared prompt construction so training, evaluation and the demo all present
retrieved context to the model in exactly the same format.
"""

RAG_TEMPLATE = (
    "Answer the question using the excerpts from ML papers below. "
    "Be concise and accurate.\n\n"
    "Excerpts:\n{context}\n\n"
    "Question: {question}"
)


def build_rag_user_message(question: str, retriever, k: int = 3, max_chars: int = 1800) -> str:
    """Retrieve top-k chunks for `question` and wrap them into one user message.

    Each chunk is ~500 words (~3000 chars) from ingest.py; we truncate to
    `max_chars` so 3 chunks + question + answer fit comfortably in 2048 tokens.
    """
    hits = retriever.retrieve(question, top_k=k)
    parts = [f"[Source: {h['paper']}]\n{h['text'][:max_chars]}" for h in hits]
    return RAG_TEMPLATE.format(context="\n\n".join(parts), question=question)
