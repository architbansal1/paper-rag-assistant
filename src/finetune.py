"""
LoRA / QLoRA fine-tune a small open model on the paper Q&A pairs.

Changes vs. the first version:
  * Loss is computed on the ASSISTANT TOKENS ONLY (prompt/completion format),
    instead of on the whole chat-templated string.
  * Training data = data/qa_pairs.jsonl + data/qa_pairs_extra.jsonl.
  * CLI flags for seed / epochs / lr / output dir, so several seeds can be run
    with identical settings.
  * Optional --rag: train on prompts that contain retrieved excerpts, in the
    same format eval.py --rag uses (see src/rag_prompt.py).

Designed to run on a free-tier Kaggle/Colab T4 (16GB).
Logs to Weights & Biases if WANDB_API_KEY is set; otherwise local only.
"""
import os
import json
import inspect
import argparse
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATHS = [
    os.path.join(HERE, "..", "data", "qa_pairs.jsonl"),
    os.path.join(HERE, "..", "data", "qa_pairs_extra.jsonl"),
]
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "..", "models", "lora-adapter")

USE_WANDB = bool(os.environ.get("WANDB_API_KEY"))


def patch_trl_kernel_for_old_gpus():
    """Work around a TRL crash on GPUs with compute capability < 8.0 (e.g. T4).

    trl/kernels/chunked_logprob.py calls torch.addmm(..., out_dtype=float32)
    with bf16 inputs whenever PyTorch >= 2.8, without checking the GPU. On a
    T4 (capability 7.5) that raises:
      RuntimeError: gemm input type at::BFloat16 and output type float is only
      supported for CUDA devices with compute capability 8.0 or higher
    We swap in the plain fp32 upcast path, which is what TRL itself falls back
    to on other setups. No-op on newer GPUs or if the module doesn't exist.
    """
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] >= 8:
        return
    try:
        import trl.kernels.chunked_logprob as clp
    except Exception:
        return

    def _addmm_fp32_compat(acc, a, b):
        acc.addmm_(a.float(), b.float())

    if hasattr(clp, "_addmm_fp32"):
        clp._addmm_fp32 = _addmm_fp32_compat
        print("[patch] using fp32 upcast addmm in trl chunked_logprob (GPU capability < 8.0)")


def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def build_lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_examples(rag: bool, k: int) -> Dataset:
    """Read every available Q&A file and convert to prompt/completion format.

    prompt     = [user message]       (optionally with retrieved excerpts)
    completion = [assistant message]
    TRL then applies the chat template and masks the prompt, so the loss only
    covers the assistant's answer.
    """
    retriever = None
    if rag:
        from retrieve import Retriever
        from rag_prompt import build_rag_user_message
        retriever = Retriever()

    rows = []
    for path in DATA_PATHS:
        if not os.path.exists(path):
            print(f"[data] skipping missing file {os.path.basename(path)}")
            continue
        n_before = len(rows)
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                msgs = json.loads(line)["messages"]
                question, answer = msgs[0]["content"], msgs[1]["content"]
                if rag:
                    question = build_rag_user_message(question, retriever, k=k)
                rows.append({
                    "prompt": [{"role": "user", "content": question}],
                    "completion": [{"role": "assistant", "content": answer}],
                })
        print(f"[data] {os.path.basename(path)}: {len(rows) - n_before} examples")
    print(f"[data] total training examples: {len(rows)}")
    return Dataset.from_list(rows)


def build_training_args(args):
    # SFTConfig's accepted kwargs drift across trl versions, so only pass
    # what the installed version supports.
    params = inspect.signature(SFTConfig.__init__).parameters
    max_len = 2048 if args.rag else 1024

    kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="no",
        seed=args.seed,
        bf16=torch.cuda.is_available(),
        report_to="wandb" if USE_WANDB else "none",
        run_name=f"paper-rag-lora-{'rag' if args.rag else 'closed'}-s{args.seed}" if USE_WANDB else None,
    )

    if "max_length" in params:
        kwargs["max_length"] = max_len
    elif "max_seq_length" in params:
        kwargs["max_seq_length"] = max_len

    # Plain "nll" avoids the chunked_nll crash on some transformers versions.
    if "loss_type" in params:
        kwargs["loss_type"] = "nll"

    # Loss on completion tokens only (this is already the default for
    # prompt/completion data in recent trl, but be explicit when supported).
    if "completion_only_loss" in params:
        kwargs["completion_only_loss"] = True

    return SFTConfig(**kwargs)


def build_trainer(model, tokenizer, dataset, training_args):
    params = inspect.signature(SFTTrainer.__init__).parameters
    tokenizer_kwarg = "processing_class" if "processing_class" in params else "tokenizer"
    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        **{tokenizer_kwarg: tokenizer},
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rag", action="store_true", help="train on prompts that include retrieved excerpts")
    p.add_argument("--k", type=int, default=3, help="number of retrieved chunks when --rag")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[config] seed={args.seed} epochs={args.epochs} lr={args.lr} rag={args.rag} -> {args.output_dir}")
    set_seed(args.seed)
    patch_trl_kernel_for_old_gpus()

    dataset = load_examples(args.rag, args.k)  # before the model, so retrieval gets the GPU/CPU to itself

    model, tokenizer = load_model_and_tokenizer()
    model = get_peft_model(model, build_lora_config())
    model.print_trainable_parameters()

    training_args = build_training_args(args)
    trainer = build_trainer(model, tokenizer, dataset, training_args)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
