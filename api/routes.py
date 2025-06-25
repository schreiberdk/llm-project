from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from model_logic.schemas import TextRequest, ChatRequest
from model_logic.responses import TextResponse, ChatResponse, StatusResponse
from services.generation_service import GenerationService
from services.model_service import model_service

router = APIRouter()

@router.get("/", response_model=StatusResponse)
async def get_status():
    """Get API status and model information"""
    return StatusResponse(
        status="Up and running",
        model_loaded=model_service.is_loaded,
        device=model_service.device
    )

@router.post("/load-model")
async def load_model():
    """Load the model"""
    if model_service.is_loaded:
        return {"message": "Model already loaded"}

    success = model_service.load_model()
    if success:
        return {"message": "Model loaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to load model")

@router.post("/unload-model")
async def unload_model():
    """Unload the model"""
    model_service.unload_model()
    return {"message": "Model unloaded"}

@router.get("/model-info")
async def get_model_info():
    """Get model information"""
    return model_service.get_model_info()

@router.post("/generate", response_model=TextResponse)
async def generate_text(request: TextRequest):
    """Generate text with full parameter control"""
    try:
        result = GenerationService.generate_text(request)
        return TextResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-stream")
async def generate_text_stream(request: TextRequest):
    """Stream text generation"""
    try:
        return StreamingResponse(
            GenerationService.generate_text_stream(request),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def simple_chat(request: ChatRequest):
    """Simple chat interface"""
    try:
        result = GenerationService.simple_chat(request)
        return ChatResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
