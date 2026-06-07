# tests/test_api.py
import pytest
import httpx

BASE_URL = "http://localhost:8000"
API_KEY = "bottrainer_secret_key_123"
HEADERS = {"X-Api-Key": API_KEY}


def test_health():
    r = httpx.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_intents():
    r = httpx.get(f"{BASE_URL}/intents", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["total"] == 8


def test_predict_valid():
    r = httpx.post(
        f"{BASE_URL}/predict",
        headers=HEADERS,
        json={"text": "Book a flight to Delhi"}
    )
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "book_flight"
    assert "similarity_score" in data


def test_predict_no_auth():
    r = httpx.post(
        f"{BASE_URL}/predict",
        json={"text": "test"}
    )
    assert r.status_code == 422


def test_metrics():
    r = httpx.get(f"{BASE_URL}/metrics", headers=HEADERS)
    assert r.status_code == 200
    assert "total_queries" in r.json()