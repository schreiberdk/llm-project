import torch
import time
import json
from typing import Generator, Dict, Any
from model_logic.schemas import TextRequest, ChatRequest
from services.model_service import model_service
import logging

logger = logging.getLogger(__name__)

class GenerationService:

    @staticmethod
    def generate_text(request: TextRequest) -> Dict[str, Any]:
        """Generate text using the loaded model"""
        if not model_service.is_loaded:
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        try:
            # Tokenize input
            inputs = model_service.tokenizer.encode(
                request.text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model_service.device)

            # Prepare generation parameters
            gen_kwargs = {
                "max_length": len(inputs[0]) + request.max_length,
                "temperature": request.temperature,
                "do_sample": request.do_sample,
                "pad_token_id": model_service.tokenizer.pad_token_id,
                "eos_token_id": model_service.tokenizer.eos_token_id,
            }

            # Add optional parameters
            if request.top_p is not None:
                gen_kwargs["top_p"] = request.top_p
            if request.top_k is not None:
                gen_kwargs["top_k"] = request.top_k

            # Generate
            with torch.no_grad():
                outputs = model_service.model.generate(inputs, **gen_kwargs)

            # Decode
            generated_text = model_service.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_text = generated_text[len(request.text):].strip()

            generation_time = time.time() - start_time

            return {
                "generated_text": response_text,
                "input_text": request.text,
                "generation_time": generation_time
            }

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise RuntimeError(f"Generation failed: {str(e)}")

    @staticmethod
    def generate_text_stream(request: TextRequest) -> Generator[str, None, None]:
        """Generate text with streaming response"""
        if not model_service.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            # Tokenize input
            inputs = model_service.tokenizer.encode(
                request.text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model_service.device)

            # Generate tokens one by one
            generated = inputs

            for _ in range(request.max_length):
                with torch.no_grad():
                    outputs = model_service.model(generated)
                    next_token_logits = outputs.logits[0, -1, :]

                    if request.do_sample:
                        next_token_logits = next_token_logits / request.temperature
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)
                    new_text = model_service.tokenizer.decode(next_token, skip_special_tokens=True)

                    yield f"data: {json.dumps({'token': new_text})}\n\n"

                    if next_token.item() == model_service.tokenizer.eos_token_id:
                        break

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            logger.error(f"Streaming generation failed: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    @staticmethod
    def simple_chat(request: ChatRequest) -> Dict[str, str]:
        """Simple chat interface"""
        if not model_service.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            inputs = model_service.tokenizer.encode(
                request.text,
                return_tensors="pt"
            ).to(model_service.device)

            with torch.no_grad():
                outputs = model_service.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + request.max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=model_service.tokenizer.pad_token_id,
                    eos_token_id=model_service.tokenizer.eos_token_id,
                )

            generated_text = model_service.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_text = generated_text[len(request.text):].strip()

            return {"response": response_text}

        except Exception as e:
            logger.error(f"Chat failed: {str(e)}")
            raise RuntimeError(f"Chat failed: {str(e)}")
