from pydantic import BaseModel, Field
from typing import Optional

class TextRequest(BaseModel):
    text: str = Field(..., description="Input text to generate from")
    max_length: int = Field(default=100, ge=1, le=500, description="Maximum length of generated text")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Sampling temperature")
    do_sample: bool = Field(default=True, description="Whether to use sampling")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: Optional[int] = Field(default=None, ge=1, description="Top-k sampling parameter")

class ChatRequest(BaseModel):
    text: str
    max_length: Optional[int] = 50
