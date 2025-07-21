import os
from dotenv import load_dotenv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from typing import Dict, List, Tuple



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


def build_prompt_with_history(session_id: str, user_question: str, context: str, chat_memory: Dict[str, List[Tuple[str, str]]]) -> str:
    history = chat_memory.get(session_id, [])

    # Build chat format
    dialogue = "<|system|>\nYou are a helpful and accurate medical assistant.\n\n"
    dialogue += f"Context:\n{context}\n\n"

    for user_msg, assistant_msg in history:
        dialogue += f"<|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}\n"

    # Append current question
    dialogue += f"<|user|>\n{user_question}\n<|assistant|>"

    return dialogue
