import pytest
import os
from dotenv import load_dotenv
from ml_logic.model_logic import load_tl_model, build_prompt_with_history, StopOnTokens
from transformers import StoppingCriteriaList

load_dotenv()

MAX_LENGTH_LIMIT = int(os.getenv("MAX_LENGTH_LIMIT", 256))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.8))
TOP_P = float(os.getenv("TOP_P", 0.7))
REP_PENALTY = float(os.getenv("REP_PENALTY", 1.1))

def test_model_loads():
    tokenizer, model, device = load_tl_model()
    assert tokenizer is not None
    assert model is not None
    assert device in ["cpu", "cuda"]

def test_model_output_basic():
    tokenizer, model, device = load_tl_model()
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=16)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert len(response) > 0  # Should produce some text

def test_model_output_no_roleplay():
    """
    Test that the model generates plain text and does not
    include human/assistant roleplay tokens.
    """
    tokenizer, model, device = load_tl_model()

    prompt = "Hi, I am feeling a bit unwell today"

    # Tokenize and move to correct device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Stop tokens
    stop_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in ["<|user|>", "<|system|>"]]
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])

    # Generate a short response
    num_samples = 10
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_LENGTH_LIMIT,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REP_PENALTY,
        stopping_criteria = stopping_criteria,
        num_return_sequences=num_samples
    )

    forbidden_tokens = ["<|user|>", "<|assistant|>", "Human:", "Assistant:"]

    # Check all sampled outputs
    for i, output in enumerate(outputs):
        response = tokenizer.decode(output, skip_special_tokens=True).strip()

        # Assert some text is produced
        assert len(response) > 0, f"Sample {i} produced empty response."

        # Assert no roleplay tokens present
        for token in forbidden_tokens:
            assert token not in response, f"Sample {i} contains roleplay token: {token}"
