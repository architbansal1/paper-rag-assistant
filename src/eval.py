"""
Evaluate the base model or a LoRA adapter on the held-out eval set
(eval/eval_set.jsonl + eval/eval_set_extra.jsonl; disjoint from training data).

Per question we record:
  * cosine similarity between the answer and the reference (MiniLM judge model)
  * a refusal flag, so a refusal counts as a failure instead of a quiet low score
  * the full answer, so transcripts can be read or LLM-graded (src/judge.py)

Each run is written to eval/runs/<tag>.json; src/report.py aggregates runs.

Usage:
  python src/eval.py --tag base_closed
  python src/eval.py --rag --tag base_rag
  python src/eval.py --adapter models/ft_closed_s1 --tag ft_closed__s1
  python src/eval.py --adapter models/ft_rag_s1 --rag --tag ft_rag__s1
"""
import os
import re
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EVAL_PATHS = [
    os.path.join(HERE, "..", "eval", "eval_set.jsonl"),
    os.path.join(HERE, "..", "eval", "eval_set_extra.jsonl"),
]
RUNS_DIR = os.path.join(HERE, "..", "eval", "runs")
JUDGE_EMBED_MODEL = "all-MiniLM-L6-v2"

_REFUSAL_START = re.compile(
    r"^\s*(i'?m sorry|i am sorry|sorry|i apologi[sz]e|i can(?:no|')t (?:assist|help|answer|provide)|i cannot (?:assist|help|answer|provide)|i'?m unable|i am unable)",
    re.IGNORECASE,
)
_REFUSAL_ANY = re.compile(
    r"(can'?t|cannot|unable to|not able to) (assist|help)( with)?( that| this)?", re.IGNORECASE
)


def is_refusal(answer: str) -> bool:
    """Heuristic: starts with a refusal phrase, or is short and contains one."""
    if _REFUSAL_START.search(answer):
        return True
    return len(answer.split()) < 30 and bool(_REFUSAL_ANY.search(answer))


def load_eval_set():
    items = []
    for path in EVAL_PATHS:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            items += [json.loads(line) for line in f if line.strip()]
    return items


def generate(model, tokenizer, user_message: str, max_new_tokens=200) -> str:
    messages = [{"role": "user", "content": user_message}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy, for reproducible eval
        )
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def run_eval(adapter_path, rag: bool, k: int, tag: str, max_new_tokens: int):
    retriever = None
    if rag:
        from retrieve import Retriever
        from rag_prompt import build_rag_user_message
        retriever = Retriever()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    label = "base"
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        label = "fine-tuned"
    model.eval()

    judge = SentenceTransformer(JUDGE_EMBED_MODEL)
    eval_set = load_eval_set()

    results, sims, refusals = [], [], 0
    for item in eval_set:
        question = item["messages"][0]["content"]
        reference = item["messages"][1]["content"]
        user_message = build_rag_user_message(question, retriever, k=k) if rag else question
        answer = generate(model, tokenizer, user_message, max_new_tokens=max_new_tokens)

        sim = util.cos_sim(judge.encode(answer), judge.encode(reference)).item()
        refused = is_refusal(answer)
        sims.append(sim)
        refusals += int(refused)
        results.append({
            "question": question,
            "paper": item.get("paper"),
            "reference": reference,
            "answer": answer,
            "similarity": round(sim, 4),
            "refusal": refused,
        })
        flag = "  <-- REFUSAL" if refused else ""
        print(f"[{tag}] sim={sim:.3f}  Q: {question[:60]}...{flag}")

    avg_sim = sum(sims) / len(sims)
    print(f"\n{tag} ({label}, rag={rag}) — n={len(sims)}  avg cosine={avg_sim:.4f}  refusals={refusals}")

    os.makedirs(RUNS_DIR, exist_ok=True)
    out_path = os.path.join(RUNS_DIR, f"{tag}.json")
    with open(out_path, "w") as f:
        json.dump({
            "tag": tag, "label": label, "adapter": adapter_path, "rag": rag, "k": k if rag else None,
            "n": len(sims), "average_similarity": avg_sim, "refusals": refusals, "results": results,
        }, f, indent=2)
    print(f"Saved {out_path}")
    print(
        "Note: cosine similarity is a proxy; read the transcripts or run src/judge.py "
        "before trusting any aggregate."
    )
    return avg_sim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter path; omit for the base model")
    parser.add_argument("--rag", action="store_true", help="put retrieved excerpts in the prompt")
    parser.add_argument("--k", type=int, default=3, help="retrieved chunks when --rag")
    parser.add_argument("--tag", type=str, default=None, help="run name; also the output filename")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()
    tag = args.tag or (("ft" if args.adapter else "base") + ("_rag" if args.rag else "_closed"))
    run_eval(args.adapter, args.rag, args.k, tag, args.max_new_tokens)
