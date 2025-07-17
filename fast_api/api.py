import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from ml_logic.tiny_llama_logic import load_tl_model

# Initialize API
app = FastAPI()

# Load model on startup
tokenizer, model, device = load_tl_model()
app.state.tokenizer = tokenizer
app.state.model = model
app.state.device = device


# Test route
@app.get("/")
def index():
    return {"status": "Up and running."}

# Request model
class PromptRequest(BaseModel):
    prompt: str

# Post route
@app.post('/prompt_model')
async def prompt_model(request: PromptRequest):
    prompt = "### Human: " + request.prompt + "\n### Assistant:"
    tokenizer = app.state.tokenizer
    model = app.state.model

    # Fetch model parameters
    MAX_LENGTH_LIMIT = int(os.getenv("MAX_LENGTH_LIMIT", 256))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.8))
    TOP_P = float(os.getenv("TOP_P", 0.7))
    REP_PENALTY = float(os.getenv("REP_PENALTY", 1.1))

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
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REP_PENALTY
        )

        # Decode and return
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from output
        if response.startswith(prompt):
            response = response[len(prompt):].lstrip()

        return {"response": response}

    except Exception as e:
        return {"error": str(e)}
