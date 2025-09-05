# Use official Python 3.12.1 image
FROM python:3.12.1-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Optionally pre-download Hugging Face model during build
#RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0', cache_dir='/model_cache')"

# Expose FastAPI default port
EXPOSE 8001

# Start the FastAPI server using uvicorn
CMD ["uvicorn", "fast_api.api:app", "--host", "0.0.0.0", "--port", "8001"]
