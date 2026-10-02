"""
Merge the LoRA adapter into the base model, then quantize for serving.

This is the step that most directly extends the author's existing
professional quantization work (AIMET/PTQ on CNNs for Galaxy devices) into
the LLM domain — same bit-width/accuracy tradeoff thinking, different
model class and tooling.

Two paths, pick based on what's installed:
  1. GGUF via llama.cpp (simplest, runs on CPU too — good for a laptop demo)
  2. AWQ via autoawq (more involved, closer to production LLM serving)

This script handles the merge step and documents both quantization paths;
actual GGUF conversion uses llama.cpp's convert script (see README).
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "lora-adapter")
MERGED_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "merged-fp16")


def merge_adapter():
    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"Loading LoRA adapter from {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("Merging adapter into base weights...")
    merged_model = model.merge_and_unload()

    os.makedirs(MERGED_PATH, exist_ok=True)
    merged_model.save_pretrained(MERGED_PATH)
    tokenizer.save_pretrained(MERGED_PATH)
    print(f"Merged fp16 model saved to {MERGED_PATH}")
    print(
        "\nNext step — quantize this merged model. Two options:\n"
        "\n"
        "Option A (GGUF, simplest, CPU-friendly):\n"
        "  git clone https://github.com/ggerganov/llama.cpp\n"
        "  cd llama.cpp && pip install -r requirements.txt\n"
        f"  python convert_hf_to_gguf.py {MERGED_PATH} --outfile paper_rag_q4.gguf --outtype q4_0\n"
        "  ./llama-server -m paper_rag_q4.gguf -c 2048   # serves an OpenAI-compatible API\n"
        "\n"
        "Option B (AWQ, closer to production LLM serving):\n"
        "  pip install autoawq\n"
        "  # quantize with AutoAWQ's calibration-based quantize() call,\n"
        "  # then serve with vLLM: vllm serve <awq-model-path> --quantization awq\n"
    )


if __name__ == "__main__":
    merge_adapter()
