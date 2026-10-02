# Paper-RAG: A RAG + Fine-Tuned Assistant for ML Papers

A small end-to-end GenAI project covering retrieval-augmented generation,
LoRA fine-tuning, evaluation, and LLM quantization — built on a curated
set of ML papers chosen to match my existing background in model
quantization (AIMET/PTQ at Samsung) and few-shot/metric learning
(personalized gesture recognition).

## What this covers

| Stage | Technique | Script |
|---|---|---|
| Retrieval | Chunking, embeddings, vector search (Chroma) | `src/ingest.py`, `src/retrieve.py` |
| Fine-tuning | LoRA (PEFT + TRL) on Qwen2.5-1.5B-Instruct | `src/finetune.py` |
| Evaluation | Held-out eval set, disjoint from training data | `src/eval.py` |
| Quantization & serving | Merge + GGUF/AWQ quantization | `src/serve.py` |
| Tool use | Minimal function-calling example | `src/app.py` |

## Why this scope, not a generic chatbot

The paper set (`data/papers_list.md`) deliberately includes quantization
papers (GPTQ, SmoothQuant) and few-shot learning papers (Prototypical
Networks, Matching Networks) alongside core GenAI fundamentals
(Attention, LoRA, RAG). This keeps the project's content area coherent
with the rest of my experience rather than reading as an unrelated
side project.

## Setup

```bash
pip install -r requirements.txt
python src/download_papers.py   # downloads 8 papers from arXiv
python src/ingest.py            # chunks + embeds into Chroma
```

## Fine-tuning

Designed to run on a free-tier Colab/Kaggle T4 GPU.

```bash
python src/finetune.py
```

Set `WANDB_API_KEY` in the environment to log training runs to
Weights & Biases.

## Evaluation

The eval set (`eval/eval_set.jsonl`) is intentionally disjoint from the
training data (`data/qa_pairs.jsonl`) — same held-out discipline used in
my Samsung Ring gesture-recognition project (train/tune on one split,
evaluate on data the model never saw).

```bash
python src/eval.py                              # base model
python src/eval.py --adapter models/lora-adapter # fine-tuned model
```

Reports average embedding-similarity-to-reference as an automatic proxy
metric, plus a saved transcript for manual or LLM-as-judge review — a
single aggregate number isn't trusted on its own.

## Quantization & serving

```bash
python src/serve.py   # merges LoRA adapter, prints next steps for GGUF/AWQ quantization
```

## Results

_Fill in after running eval.py on both base and fine-tuned models:_

| Model | Avg. cosine similarity to reference |
|---|---|
| Base (Qwen2.5-1.5B-Instruct) | TBD |
| LoRA fine-tuned | TBD |

## What I'd extend next

- Replace the cosine-similarity proxy metric with a proper LLM-as-judge pass
- Add a few-shot ablation on the eval set, the way I did for the prototypical
  network project (does accuracy hold with fewer fine-tuning examples?)
- Swap the single `list_papers` tool for a small set of real tools
  (e.g., "fetch latest arXiv version", "summarize a specific section")
