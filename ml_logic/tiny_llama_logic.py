import os
from dotenv import load_dotenv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load .env variables
load_dotenv()

# Fetch MODEL_PATH from environment
MODEL_PATH = os.getenv("TINY_LLAMA_PATH", "models/TinyLlama")  # fallback if not set
DEVICE = os.getenv("DEVICE", "cpu")  # fallback if not set

def load_tl_model(model_path: str = None):
    model_path = model_path or MODEL_PATH

    device = DEVICE

    tokenizer = AutoTokenizer.from_pretrained(
                    model_path
                )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    return tokenizer, model, device
