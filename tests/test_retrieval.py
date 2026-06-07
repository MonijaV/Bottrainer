import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.embeddings import EmbeddingModel
from retrieval.faiss_retriever import FAISSRetriever


@pytest.fixture(scope="module")
def retriever():
    embedding_model = EmbeddingModel()
    r = FAISSRetriever(embedding_model)
    r.load(
        "retrieval/faiss_index/index.faiss",
        "retrieval/faiss_index/metadata.json"
    )
    return r


def test_index_loads_correctly(retriever):
    assert retriever.index is not None
    assert retriever.index.ntotal > 0
    print(f"\nIndex size: {retriever.index.ntotal} vectors")


def test_search_returns_correct_count(retriever):
    results = retriever.search("Book a flight to Delhi", k=3)
    assert len(results) == 3


def test_similarity_scores_valid_range(retriever):
    results = retriever.search("Order me some food", k=3)
    for r in results:
        assert 0.0 <= r.similarity_score <= 1.0


def test_flight_query_returns_flight_intent(retriever):
    results = retriever.search("I need to fly to Mumbai tomorrow", k=3)
    top_intent = results[0].intent
    assert top_intent == "book_flight"


def test_food_query_returns_food_intent(retriever):
    results = retriever.search("Order me a pizza please", k=3)
    top_intent = results[0].intent
    assert top_intent == "order_food"


def test_weather_query_returns_weather_intent(retriever):
    results = retriever.search("Will it rain tomorrow?", k=3)
    top_intent = results[0].intent
    assert top_intent == "check_weather"


def test_high_similarity_for_clear_query(retriever):
    results = retriever.search("Book a flight to Delhi", k=3)
    assert results[0].similarity_score >= 0.45


def test_is_query_clear_true_for_known_intent(retriever):
    results = retriever.search("Cancel my booking", k=3)
    assert retriever.is_query_clear(results) == True


def test_results_have_required_fields(retriever):
    results = retriever.search("Play some music", k=3)
    for r in results:
        assert hasattr(r, "text")
        assert hasattr(r, "intent")
        assert hasattr(r, "similarity_score")
        assert isinstance(r.text, str)
        assert isinstance(r.intent, str)
        assert isinstance(r.similarity_score, float)