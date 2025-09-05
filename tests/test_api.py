import pytest
from fastapi.testclient import TestClient
from fast_api.api import app

client = TestClient(app)

def test_index_route():
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()

def test_prompt_model_route():
    with TestClient(app) as client:
        response = client.post("/prompt_model", json={"prompt": "Hello"})
        assert response.status_code == 200
        assert "response" in response.json()
        assert len(response.json()["response"]) > 0
