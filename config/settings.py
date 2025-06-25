from pydantic import BaseSettings

class Settings(BaseSettings):
    model_path: str = "../models/TinyLlama"
    device: str = "cuda"
    max_length_limit: int = 512
    temperature_min: float = 0.1
    temperature_max: float = 2.0

    class Config:
        env_file = ".env"

settings = Settings()
