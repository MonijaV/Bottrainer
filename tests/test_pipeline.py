import pytest
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipeline import NLUPipeline
from validation.validator import FinalResponse


@pytest.fixture(scope="module")
def pipeline():
    return NLUPipeline()


@pytest.mark.asyncio
async def test_pipeline_returns_final_response(pipeline):
    result = await pipeline.predict("Book a flight to Delhi")
    assert isinstance(result, FinalResponse)


@pytest.mark.asyncio
async def test_pipeline_correct_intent(pipeline):
    result = await pipeline.predict("Book a flight to Delhi tomorrow")
    assert result.intent == "book_flight"


@pytest.mark.asyncio
async def test_pipeline_extracts_entities(pipeline):
    result = await pipeline.predict("Book a flight to Delhi tomorrow")
    assert "location" in result.entities or "date" in result.entities


@pytest.mark.asyncio
async def test_pipeline_empty_input(pipeline):
    result = await pipeline.predict("")
    assert result.intent == "unclear"
    assert result.is_clear == False


@pytest.mark.asyncio
async def test_pipeline_has_latency_fields(pipeline):
    result = await pipeline.predict("Order me a pizza")
    assert result.retrieval_time_ms > 0
    assert result.llm_time_ms > 0
    assert result.total_time_ms > 0


@pytest.mark.asyncio
async def test_pipeline_has_retrieved_examples(pipeline):
    result = await pipeline.predict("Check weather in Mumbai")
    assert len(result.retrieved_examples) > 0
    assert all(isinstance(e, str) for e in result.retrieved_examples)


@pytest.mark.asyncio
async def test_pipeline_out_of_scope(pipeline):
    result = await pipeline.predict("What is the capital of France?")
    assert result.intent in ["out_of_scope", "unclear"]


@pytest.mark.asyncio
async def test_pipeline_similarity_score_valid(pipeline):
    result = await pipeline.predict("Play some music")
    assert 0.0 <= result.similarity_score <= 1.0