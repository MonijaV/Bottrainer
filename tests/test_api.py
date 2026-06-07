import pytest
import httpx
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_URL = "http://localhost:8000"
API_KEY = "bottrainer_secret_key_123"
HEADERS = {"X-Api-Key": API_KEY}


def test_health_check():
    r = httpx.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "model" in data


def test_get_intents_with_auth():
    r = httpx.get(f"{BASE_URL}/intents", headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 8
    assert "book_flight" in data["intents"]
    assert "out_of_scope" in data["intents"]


def test_predict_valid_request():
    r = httpx.post(
        f"{BASE_URL}/predict",
        headers=HEADERS,
        json={"text": "Book a flight to Delhi"},
        timeout=30.0
    )
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "book_flight"
    assert "entities" in data
    assert "similarity_score" in data
    assert "retrieval_time_ms" in data
    assert "llm_time_ms" in data


def test_predict_without_api_key():
    r = httpx.post(
        f"{BASE_URL}/predict",
        json={"text": "test"}
    )
    assert r.status_code == 422


def test_predict_wrong_api_key():
    r = httpx.post(
        f"{BASE_URL}/predict",
        headers={"X-Api-Key": "wrong_key"},
        json={"text": "test"}
    )
    assert r.status_code == 401


def test_get_metrics():
    r = httpx.get(f"{BASE_URL}/metrics", headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert "total_queries" in data
    assert "avg_total_ms" in data
    assert "intent_distribution" in data


def test_predict_response_structure():
    r = httpx.post(
        f"{BASE_URL}/predict",
        headers=HEADERS,
        json={"text": "Order me a pizza"},
        timeout=30.0
    )
    assert r.status_code == 200
    data = r.json()
    required_fields = [
        "intent", "entities", "similarity_score",
        "is_clear", "retrieval_time_ms", "llm_time_ms",
        "total_time_ms", "retrieved_examples"
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"