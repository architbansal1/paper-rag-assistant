"""
LoRA fine-tune a small open model on data/qa_pairs.jsonl.

Designed to run on a free-tier Colab/Kaggle T4 GPU (16GB).
Model: Qwen2.5-1.5B-Instruct (small enough to fine-tune comfortably on T4).

Logs to Weights & Biases if WANDB_API_KEY is set in the environment;
otherwise falls back to local logging only.
"""
import os
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "qa_pairs.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "lora-adapter")

USE_WANDB = bool(os.environ.get("WANDB_API_KEY"))


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
    # Targets the attention projection matrices — see data/qa_pairs.jsonl's
    # own LoRA Q&A pair for why these layers specifically.
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def main():
    model, tokenizer = load_model_and_tokenizer()
    lora_config = build_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    # SFTConfig's accepted kwargs have changed across trl versions (e.g.
    # max_seq_length -> max_length). Build the base args, then add the
    # length-limit kwarg under whichever name this installed version wants.
    base_kwargs = dict(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        report_to="wandb" if USE_WANDB else "none",
        run_name="paper-rag-lora" if USE_WANDB else None,
    )
    try:
        training_args = SFTConfig(max_length=1024, **base_kwargs)
    except TypeError:
        try:
            training_args = SFTConfig(max_seq_length=1024, **base_kwargs)
        except TypeError:
            # Neither kwarg accepted on this version — fall back to default length.
            training_args = SFTConfig(**base_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
