"""
Ties everything together: retrieval + fine-tuned model + one simple tool call.

The tool call ("list_papers") is a minimal, honest example of function
calling / agentic behavior — the model can decide to call it when asked
what sources are available, rather than guessing. Kept intentionally small;
the point of this project is to cover the core mechanics, not build a
large agent framework.
"""
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from retrieve import Retriever

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "lora-adapter")
PAPERS_LIST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "papers_list.md")


def list_papers() -> str:
    """The one tool the model can call."""
    if not os.path.exists(PAPERS_LIST_PATH):
        return "No papers list found."
    with open(PAPERS_LIST_PATH) as f:
        return f.read()


TOOLS = {"list_papers": list_papers}


def load_model(use_adapter=True):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    if use_adapter and os.path.exists(ADAPTER_PATH):
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    return model, tokenizer


def answer(question: str, model, tokenizer, retriever: Retriever) -> str:
    # Simple routing: a question about "what papers"/"what sources" calls
    # the tool directly instead of going through RAG.
    if "what papers" in question.lower() or "what sources" in question.lower():
        return list_papers()

    hits = retriever.retrieve(question, top_k=4)
    context = retriever.format_context(hits)

    prompt_messages = [
        {
            "role": "system",
            "content": (
                "You answer questions about ML papers using the provided context. "
                "Cite which paper each claim comes from. If the context doesn't "
                "contain the answer, say so rather than guessing."
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What is LoRA and why is it memory-efficient?"
    print(f"Q: {question}\n")

    model, tokenizer = load_model()
    retriever = Retriever()
    print(answer(question, model, tokenizer, retriever))
