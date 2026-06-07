import faiss
import numpy as np
import json
import os
from pathlib import Path
from pydantic import BaseModel

from retrieval.embeddings import EmbeddingModel


# ── Result Model ─────────────────────────────────────────────────────────────

class RetrievalResult(BaseModel):
    text: str
    intent: str
    similarity_score: float

    class Config:
        frozen = True


# ── Index Builder ─────────────────────────────────────────────────────────────

class FAISSIndexBuilder:
    """
    Builds and saves a FAISS index from training examples.
    Run this once when data changes.
    """

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.index = None
        self.metadata = []

    def build(self, examples: list[dict]) -> None:
        """
        Build FAISS index from list of {text, intent} dicts.

        Args:
            examples: List of dicts with 'text' and 'intent' keys
        """
        print(f"Building FAISS index from {len(examples)} examples...")

        texts = [e["text"] for e in examples]
        self.metadata = [{"text": e["text"], "intent": e["intent"]}
                         for e in examples]

        # Encode all examples
        embeddings = self.embedding_model.encode(texts)
        print(f"Generated embeddings: {embeddings.shape}")

        # Build flat index using inner product
        # Since embeddings are normalized, inner product = cosine similarity
        dim = self.embedding_model.embedding_dim
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        print(f"FAISS index built with {self.index.ntotal} vectors")

    def save(self, index_path: str, metadata_path: str) -> None:
        """Save index and metadata to disk."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        faiss.write_index(self.index, index_path)
        print(f"Index saved to: {index_path}")

        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")


# ── Retriever ─────────────────────────────────────────────────────────────────

class FAISSRetriever:
    """
    Loads FAISS index and retrieves most similar examples for any query.
    Similarity score is a real cosine similarity — used for confidence.
    """

    # Threshold below which query is considered unclear
    SIMILARITY_THRESHOLD = 0.45

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.index = None
        self.metadata = []

    def load(self, index_path: str, metadata_path: str) -> None:
        """Load pre-built index from disk."""
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                f"Run build_index.py first."
            )

        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        print(f"FAISS index loaded: {self.index.ntotal} vectors")

    def search(self, query: str, k: int = 3) -> list[RetrievalResult]:
        """
        Find k most similar training examples for a query.

        Args:
            query: User input text
            k: Number of results to return

        Returns:
            List of RetrievalResult sorted by similarity (highest first)
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() first.")

        query_embedding = self.embedding_model.encode_single(query)

        # Search FAISS index
        similarities, indices = self.index.search(query_embedding, k)

        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            meta = self.metadata[idx]
            results.append(RetrievalResult(
                text=meta["text"],
                intent=meta["intent"],
                similarity_score=float(similarity)
            ))

        return results

    def is_query_clear(self, results: list[RetrievalResult]) -> bool:
        """
        Check if query is clear enough to process.
        Based on FAISS similarity score — not LLM confidence.
        """
        if not results:
            return False
        return results[0].similarity_score >= self.SIMILARITY_THRESHOLD

    def get_top_similarity(self, results: list[RetrievalResult]) -> float:
        """Get the highest similarity score from results."""
        if not results:
            return 0.0
        return results[0].similarity_score


# ── Build Script ──────────────────────────────────────────────────────────────

def build_faiss_index():
    """
    Build and save FAISS index from intents.json.
    Run this whenever training data changes.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data.data_loader import DataLoader

    print("=" * 50)
    print("Building FAISS Index")
    print("=" * 50)

    # Load data
    loader = DataLoader()
    all_examples = loader.get_all_examples_flat()
    examples_dict = [{"text": e.text, "intent": e.intent}
                     for e in all_examples]

    # Build embedding model
    embedding_model = EmbeddingModel()

    # Build index
    builder = FAISSIndexBuilder(embedding_model)
    builder.build(examples_dict)
    builder.save(
        index_path="retrieval/faiss_index/index.faiss",
        metadata_path="retrieval/faiss_index/metadata.json"
    )

    print("\n✅ FAISS index built and saved successfully")
    return embedding_model


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: Build index
    embedding_model = build_faiss_index()

    print("\n" + "=" * 50)
    print("Testing Retrieval")
    print("=" * 50)

    # Step 2: Load and test retriever
    retriever = FAISSRetriever(embedding_model)
    retriever.load(
        index_path="retrieval/faiss_index/index.faiss",
        metadata_path="retrieval/faiss_index/metadata.json"
    )

    # Test queries
    test_queries = [
        "Arrange air travel to Hyderabad this weekend",
        "I am starving get me something to eat",
        "gonna rain tomorrow?",
        "cancel my reservation please",
        "where is my stuff?",
        "don't let me forget at 8pm",
        "put on some tunes",
        "who is the prime minister of India?"
    ]

    print()
    for query in test_queries:
        results = retriever.search(query, k=3)
        top = results[0]
        clear = retriever.is_query_clear(results)

        print(f"Query: '{query}'")
        print(f"  Top match: '{top.text}'")
        print(f"  Intent: {top.intent}")
        print(f"  Similarity: {top.similarity_score:.3f}")
        print(f"  Query clear: {clear}")
        print()