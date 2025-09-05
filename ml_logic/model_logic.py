import os
from dotenv import load_dotenv

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import re
from typing import Dict, List, Tuple


# Load .env variables
load_dotenv()

# Fetch MODEL_PATH from environment
MODEL_PATH = os.getenv("TINY_LLAMA_PATH", "models/TinyLlama")  # fallback if not set
MODEL_NAME = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "True").lower() in ("true", "1", "yes")
DEVICE = os.getenv("DEVICE", "cpu")  # fallback if not set

#Load the model, from directory if local, from HF if not local
def load_tl_model(model_path: str = None):
    model_id = MODEL_PATH if LOCAL_MODEL else MODEL_NAME
    device = DEVICE

    if not LOCAL_MODEL:
        # Pre-download model from HF Hub (to cache inside container)
        snapshot_download(repo_id=MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
                    model_id
                )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
        )

    if device == "cuda" and not model.device.type == "cuda":
        model = model.to("cuda")

    return tokenizer, model, device

#Build the prompt with chat history
MAX_TURNS = int(os.getenv("MAX_TURNS", 5))
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful and accurate medical assistant."
)

ChatHistory = Dict[str, List[Tuple[str, str]]]

def build_prompt_with_history(session_id: str, user_question: str, context: str, chat_memory: ChatHistory) -> str:
    history = chat_memory.get(session_id, [])
    history = history[-MAX_TURNS:]

    # Build chat format
    dialogue = f"<|system|>\n{SYSTEM_PROMPT}\n\n"
    if context.strip():
        dialogue += f"Context:\n{context.strip()}\n\n"

    for user_msg, assistant_msg in history:
        dialogue += f"<|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}\n"

    # Append current question
    dialogue += f"<|user|>\n{user_question.strip()}\n<|assistant|>"

    return dialogue


# Stopping criteria
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        # Stop if last token matches any stop token
        return any(input_ids[0, -1] == stop_id for stop_id in self.stop_ids)
