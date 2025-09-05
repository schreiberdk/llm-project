import os
from dotenv import load_dotenv
from fastapi import FastAPI
import re
from pydantic import BaseModel
from ml_logic.model_logic import load_tl_model, build_prompt_with_history, StopOnTokens
from ml_logic.context_retrieval import retrieve_context, load_vectorstore, build_vectorstore
from ml_logic.s3_utlis import fetch_all_pdfs_from_s3
from transformers import StoppingCriteriaList
from typing import Dict, List, Tuple
from pathlib import Path

# Initialize API
app = FastAPI()

load_dotenv()
topic_1 = os.getenv("S3_FOLDER_1")

# App startup procedure
@app.on_event("startup")
def startup_event():
    print("🚀 Loading TinyLlama model...")
    tokenizer, model, device = load_tl_model()
    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.device = device

    print("📦 Fetching PDFs from S3...")
    pdf_paths = fetch_all_pdfs_from_s3(topic_1)

    print("📚 Building vectorstore...")
    build_vectorstore(pdf_folder="/tmp/pdf_data", index_path="vector_db")

    print("📦 Loading vectorstore...")
    vectorstore = load_vectorstore("vector_db")
    print(f"✅ Vectorstore loaded with {len(vectorstore.docstore._dict)} documents.")

    app.state.vectorstore = vectorstore


# Chat history
ChatHistory = Dict[str, List[Tuple[str, str]]]
chat_memory = {}

# Define test route
@app.get("/")
def index():
    return {"status": "Up and running."}

# Request model
class PromptRequest(BaseModel):
    prompt: str
    session_id: str = "default"  # could be a UUID or client token


# Post route
@app.post('/prompt_model')
async def prompt_model(request: PromptRequest):
    #New prompt structure with Langchain
    user_question = request.prompt
    session_id = request.session_id

    # Use vectorstore from app state
    vectorstore = app.state.vectorstore

    # Retrieve relevant context chunks from your PDF vectorstore
    context = retrieve_context(user_question, vectorstore)

    # Build the TinyLlama chat-style prompt with injected context
    prompt = build_prompt_with_history(session_id, user_question, context, chat_memory)

    tokenizer = app.state.tokenizer
    model = app.state.model
    device = app.state.device

    # Fetch model parameters
    MAX_LENGTH_LIMIT = int(os.getenv("MAX_LENGTH_LIMIT", 256))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.8))
    TOP_P = float(os.getenv("TOP_P", 0.7))
    REP_PENALTY = float(os.getenv("REP_PENALTY", 1.1))

    # Stop tokens
    stop_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in ["<|user|>", "<|system|>"]]
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])

    # Check if parameters are reasonable
    assert 0 <= TEMPERATURE <= 1, "TEMPERATURE must be between 0 and 1"
    assert 0 <= TOP_P <= 1, "TOP_P must be between 0 and 1"
    assert MAX_LENGTH_LIMIT > 0, "MAX_LENGTH_LIMIT must be positive"

    try:
        # Tokenize input and move to the correct device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate output
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_LENGTH_LIMIT,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REP_PENALTY,
            stopping_criteria = stopping_criteria
        )

        # Decode and return
        output_tokens = outputs[0]
        input_length = inputs["input_ids"].shape[1]
        new_tokens = output_tokens[input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Clean any prefixes just in case
        response = re.sub(r'^(Assistant:|Human:)\s*', '', response)

        # Truncate at known stop sequence
        stop_sequence = "###"
        if stop_sequence in response:
            response = response.split(stop_sequence)[0].strip()

        chat_memory.setdefault(session_id, []).append((user_question, response))

        return {"response": response}

    except Exception as e:
        return {"error": str(e)}
