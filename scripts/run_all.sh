#!/usr/bin/env bash
# Full experiment: base vs fine-tuned, closed-book and RAG, 3 seeds each.
# Run from the repo root after download_papers.py and ingest.py.
# Fixed config (epochs=3, lr=2e-4) is set in finetune.py defaults and is NOT
# tuned against the eval set.
#
# Resumable: any eval run whose eval/runs/<tag>.json exists, and any adapter
# that is already saved, is skipped. If it fails midway, fix the problem and
# just run this script again.
set -e
mkdir -p eval/runs models

run_eval() {            # run_eval <tag> <eval.py args...>
  local tag=$1; shift
  if [ -f "eval/runs/$tag.json" ]; then echo "[skip] eval $tag"; return; fi
  python src/eval.py "$@" --tag "$tag"
}

run_train() {           # run_train <output-dir> <finetune.py args...>
  local out=$1; shift
  if [ -f "$out/adapter_model.safetensors" ]; then echo "[skip] train $out"; return; fi
  python src/finetune.py "$@" --output-dir "$out"
}

run_eval base_closed
run_eval base_rag --rag

for SEED in 1 2 3; do
  run_train models/ft_closed_s$SEED --seed $SEED
  run_eval ft_closed__s$SEED --adapter models/ft_closed_s$SEED

  run_train models/ft_rag_s$SEED --rag --seed $SEED
  run_eval ft_rag__s$SEED --adapter models/ft_rag_s$SEED --rag
done

python src/report.py
