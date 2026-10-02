"""
Evaluate base vs. LoRA fine-tuned model on the held-out eval set
(eval/eval_set.jsonl — disjoint from training data/qa_pairs.jsonl).

Two metrics, matching the "don't trust one aggregate number" lesson from
the Ring project:
  1. Embedding cosine similarity to the reference answer (automatic, cheap,
     run every time).
  2. A side-by-side transcript saved for manual / LLM-as-judge review,
     since cosine similarity alone can't catch factual correctness.

Usage:
  python src/eval.py --adapter models/lora-adapter
"""
import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EVAL_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "eval_set.jsonl")
JUDGE_EMBED_MODEL = "all-MiniLM-L6-v2"


def load_eval_set():
    with open(EVAL_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def generate(model, tokenizer, question: str, max_new_tokens=200) -> str:
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy, for reproducible eval
    )
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def run_eval(adapter_path: str | None):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    label = "base"
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        label = "fine-tuned"

    judge = SentenceTransformer(JUDGE_EMBED_MODEL)
    eval_set = load_eval_set()

    results = []
    sims = []
    for item in eval_set:
        question = item["messages"][0]["content"]
        reference = item["messages"][1]["content"]
        answer = generate(model, tokenizer, question)

        sim = util.cos_sim(judge.encode(answer), judge.encode(reference)).item()
        sims.append(sim)
        results.append(
            {"question": question, "reference": reference, "answer": answer, "similarity": round(sim, 4)}
        )
        print(f"[{label}] sim={sim:.3f}  Q: {question[:60]}...")

    avg_sim = sum(sims) / len(sims)
    print(f"\n{label} model — average cosine similarity to reference: {avg_sim:.4f}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "eval", f"results_{label}.json")
    with open(out_path, "w") as f:
        json.dump({"label": label, "average_similarity": avg_sim, "results": results}, f, indent=2)
    print(f"Full transcript saved to {out_path}")
    print(
        "Note: cosine similarity is a proxy metric, not ground truth — "
        "read results_*.json manually (or run an LLM-as-judge pass) before "
        "trusting the aggregate number, same as you would with any single eval metric."
    )
    return avg_sim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter; omit to eval base model")
    args = parser.parse_args()
    run_eval(args.adapter)
