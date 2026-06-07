import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wraps Sentence Transformers to convert text into vector embeddings.
    Uses M1 GPU (MPS) automatically when available.
    Model: all-MiniLM-L6-v2
    - Fast and lightweight
    - Strong semantic similarity performance
    - Runs well on CPU and M1 MPS
    - Output: 384-dimensional vectors
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        # Automatically use M1 GPU if available
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Loading embedding model on device: {self.device}")
        self.model = SentenceTransformer(self.MODEL_NAME, device=self.device)
        print(f"Embedding model loaded: {self.MODEL_NAME}")

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Convert list of texts to normalized embedding vectors.
        Normalization enables cosine similarity via dot product.

        Args:
            texts: List of strings to encode

        Returns:
            numpy array of shape (len(texts), 384)
        """
        if not texts:
            raise ValueError("Cannot encode empty list of texts")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Critical for cosine similarity
            show_progress_bar=False
        )
        return embeddings.astype("float32")

    def encode_single(self, text: str) -> np.ndarray:
        """
        Convenience method for encoding a single string.

        Returns:
            numpy array of shape (1, 384)
        """
        return self.encode([text])

    @property
    def embedding_dim(self) -> int:
        return 384


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = EmbeddingModel()

    # Test basic encoding
    texts = [
        "Book a flight to Delhi",
        "Reserve airfare to Mumbai",
        "Order a pizza please",
        "What is the weather today?"
    ]

    embeddings = model.encode(texts)
    print(f"\nEncoded {len(texts)} texts")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")

    # Test similarity — similar sentences should have high score
    from numpy.linalg import norm

    def cosine_similarity(a, b):
        # Since embeddings are normalized, dot product = cosine similarity
        return float(np.dot(a, b))

    sim_flight_airfare = cosine_similarity(embeddings[0], embeddings[1])
    sim_flight_pizza = cosine_similarity(embeddings[0], embeddings[2])
    sim_flight_weather = cosine_similarity(embeddings[0], embeddings[3])

    print(f"\nSimilarity tests:")
    print(f"  'Book flight Delhi' vs 'Reserve airfare Mumbai': {sim_flight_airfare:.3f} (should be HIGH)")
    print(f"  'Book flight Delhi' vs 'Order pizza':            {sim_flight_pizza:.3f} (should be LOW)")
    print(f"  'Book flight Delhi' vs 'Weather today':          {sim_flight_weather:.3f} (should be LOW)")

    if sim_flight_airfare > sim_flight_pizza and sim_flight_airfare > sim_flight_weather:
        print("\n✅ Similarity scores correct — semantic understanding working")
    else:
        print("\n❌ Similarity scores unexpected — check embedding model")