from fastapi import FastAPI
from api.routes import router
from services.model_service import model_service
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Text Generation API",
    description="A FastAPI service for text generation using local LLM",
    version="1.0.0"
)

# Include routes
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting up...")
    success = model_service.load_model()
    if not success:
        logger.warning("Failed to load model on startup")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Shutting down...")
    model_service.unload_model()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
