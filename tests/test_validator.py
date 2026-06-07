import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.validator import ResponseValidator, FinalResponse, VALID_INTENTS


@pytest.fixture
def validator():
    return ResponseValidator()


def test_valid_intent_passes(validator):
    r = validator.validate(
        intent="book_flight",
        entities={"location": "Delhi"},
        similarity_score=0.87,
        retrieved_examples=["Book a flight"],
        retrieval_time_ms=10.0,
        llm_time_ms=500.0,
        total_time_ms=510.0
    )
    assert r.intent == "book_flight"
    assert r.is_clear == True
    assert r.message is None


def test_unknown_intent_becomes_out_of_scope(validator):
    r = validator.validate(
        intent="completely_unknown_intent",
        entities={},
        similarity_score=0.7,
        retrieved_examples=["example"],
        retrieval_time_ms=10.0,
        llm_time_ms=500.0,
        total_time_ms=510.0
    )
    assert r.intent == "out_of_scope"


def test_low_similarity_returns_unclear(validator):
    r = validator.validate(
        intent="book_flight",
        entities={},
        similarity_score=0.20,
        retrieved_examples=["example"],
        retrieval_time_ms=10.0,
        llm_time_ms=500.0,
        total_time_ms=510.0
    )
    assert r.intent == "unclear"
    assert r.is_clear == False
    assert r.message is not None


def test_boundary_threshold_exactly_045(validator):
    r = validator.validate(
        intent="check_weather",
        entities={"location": "Mumbai"},
        similarity_score=0.45,
        retrieved_examples=["Weather in Mumbai"],
        retrieval_time_ms=10.0,
        llm_time_ms=500.0,
        total_time_ms=510.0
    )
    assert r.intent == "check_weather"
    assert r.is_clear == True


def test_empty_entities_handled(validator):
    r = validator.validate(
        intent="play_music",
        entities={},
        similarity_score=0.75,
        retrieved_examples=["Play music"],
        retrieval_time_ms=10.0,
        llm_time_ms=500.0,
        total_time_ms=510.0
    )
    assert r.entities == {}
    assert r.intent == "play_music"


def test_final_response_has_all_fields(validator):
    r = validator.validate(
        intent="order_food",
        entities={"food_item": "pizza"},
        similarity_score=0.90,
        retrieved_examples=["Order pizza"],
        retrieval_time_ms=12.0,
        llm_time_ms=450.0,
        total_time_ms=462.0
    )
    assert isinstance(r, FinalResponse)
    assert hasattr(r, "intent")
    assert hasattr(r, "entities")
    assert hasattr(r, "similarity_score")
    assert hasattr(r, "is_clear")
    assert hasattr(r, "retrieval_time_ms")
    assert hasattr(r, "llm_time_ms")
    assert hasattr(r, "total_time_ms")
    assert hasattr(r, "retrieved_examples")


def test_all_valid_intents_pass(validator):
    for intent in VALID_INTENTS:
        r = validator.validate(
            intent=intent,
            entities={},
            similarity_score=0.80,
            retrieved_examples=["example"],
            retrieval_time_ms=10.0,
            llm_time_ms=500.0,
            total_time_ms=510.0
        )
        assert r.intent == intent
        assert r.is_clear == True