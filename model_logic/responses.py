from pydantic import BaseModel

class TextResponse(BaseModel):
    generated_text: str
    input_text: str
    generation_time: float

class ChatResponse(BaseModel):
    response: str

class StatusResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
