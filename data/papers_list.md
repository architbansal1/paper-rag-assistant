# Papers used in this project

Curated to match the author's actual ML background — quantization/compression,
few-shot/metric learning, and core transformer/RAG fundamentals — so the
project's scope is coherent with the rest of the resume, not a generic
"chat with any PDF" demo.

| Short name | Title | arXiv ID | Why it's here |
|---|---|---|---|
| attention | Attention Is All You Need | 1706.03762 | Transformer fundamentals |
| lora | LoRA: Low-Rank Adaptation of Large Language Models | 2106.09685 | Fine-tuning method used in this project |
| rag | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | 2005.11401 | RAG architecture used in this project |
| protonet | Prototypical Networks for Few-shot Learning | 1703.05175 | Ties to author's gesture-recognition work |
| matching_net | Matching Networks for One Shot Learning | 1606.04080 | Ties to author's gesture-recognition work |
| gptq | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | 2210.17323 | Ties to author's on-device quantization work |
| smoothquant | SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs | 2211.10438 | LLM-specific quantization, extends author's AIMET/PTQ background |
| distillation | Distilling the Knowledge in a Neural Network | 1503.02531 | Core model-compression concept |

## Download

```bash
python src/download_papers.py
```

Downloads PDFs from arXiv into `data/papers/`.
