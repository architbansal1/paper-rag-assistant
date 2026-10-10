#!/usr/bin/env bash
# Full experiment: base vs fine-tuned, closed-book and RAG, 3 seeds each.
# Run from the repo root after download_papers.py and ingest.py.
# Fixed config (epochs=3, lr=2e-4) is set in finetune.py defaults and is NOT
# tuned against the eval set.
set -e
mkdir -p eval/runs

python src/eval.py --tag base_closed
python src/eval.py --rag --tag base_rag

for SEED in 1 2 3; do
  python src/finetune.py --seed $SEED --output-dir models/ft_closed_s$SEED
  python src/eval.py --adapter models/ft_closed_s$SEED --tag ft_closed__s$SEED

  python src/finetune.py --rag --seed $SEED --output-dir models/ft_rag_s$SEED
  python src/eval.py --adapter models/ft_rag_s$SEED --rag --tag ft_rag__s$SEED
done

python src/report.py
