import asyncio
import time
import sys
from pathlib import Path

from data.data_loader import DataLoader
from retrieval.embeddings import EmbeddingModel
from retrieval.faiss_retriever import FAISSRetriever
from prompts.prompt_builder import PromptBuilder
from llm.openai_client import LLMClient
from validation.validator import ResponseValidator, FinalResponse
from core.preprocessor import preprocess, is_valid_input
from core.logger import QueryLogger


class NLUPipeline:
    """
    Core NLU Pipeline — connects every component in sequence.

    Flow:
    User Input
        ↓ preprocess
        ↓ FAISS semantic retrieval
        ↓ Dynamic prompt construction
        ↓ Groq LLM inference
        ↓ Pydantic validation
        ↓ SQLite logging
        ↓ FinalResponse

    Why this architecture:
    Each component is independent and testable separately.
    Changing one component does not break others.
    This is how production ML pipelines are structured.
    """

    FAISS_INDEX_PATH = "retrieval/faiss_index/index.faiss"
    FAISS_METADATA_PATH = "retrieval/faiss_index/metadata.json"

    def __init__(self):
        print("Initializing NLU Pipeline...")

        # Load all components
        self.data_loader = DataLoader()
        self.embedding_model = EmbeddingModel()
        self.retriever = FAISSRetriever(self.embedding_model)
        self.prompt_builder = PromptBuilder()
        self.llm_client = LLMClient()
        self.validator = ResponseValidator()
        self.logger = QueryLogger()

        # Load FAISS index
        self.retriever.load(self.FAISS_INDEX_PATH, self.FAISS_METADATA_PATH)

        # Cache intent names
        self.valid_intents = self.data_loader.get_all_intent_names()

        print("✅ NLU Pipeline ready\n")

    async def predict(self, user_text: str) -> FinalResponse:
        """
        Main prediction method. Takes raw user input, returns FinalResponse.

        Args:
            user_text: Raw text from user

        Returns:
            FinalResponse with intent, entities, latency, and metadata
        """
        total_start = time.time()

        # Step 1 — Validate input
        if not is_valid_input(user_text):
            return FinalResponse(
                intent="unclear",
                entities={},
                similarity_score=0.0,
                is_clear=False,
                retrieval_time_ms=0.0,
                llm_time_ms=0.0,
                total_time_ms=0.0,
                retrieved_examples=[],
                message="Please enter a message."
            )

        # Step 2 — Preprocess
        cleaned_text = preprocess(user_text)

        # Step 3 — Semantic retrieval with timing
        retrieval_start = time.time()
        retrieved = self.retriever.search(cleaned_text, k=3)
        retrieval_time_ms = (time.time() - retrieval_start) * 1000

        # Step 4 — Get similarity score for confidence
        similarity_score = self.retriever.get_top_similarity(retrieved)
        retrieved_texts = [r.text for r in retrieved]

        # Step 5 — Build dynamic prompt
        prompt = self.prompt_builder.build(
            user_query=cleaned_text,
            retrieved_examples=retrieved,
            valid_intents=self.valid_intents
        )

        # Step 6 — LLM inference with timing
        llm_start = time.time()
        llm_output, llm_time_ms = await self.llm_client.predict(prompt)
        llm_time_ms = (time.time() - llm_start) * 1000

        # Step 7 — Validate response
        total_time_ms = (time.time() - total_start) * 1000
        response = self.validator.validate(
            intent=llm_output.intent,
            entities=llm_output.entities,
            similarity_score=similarity_score,
            retrieved_examples=retrieved_texts,
            retrieval_time_ms=retrieval_time_ms,
            llm_time_ms=llm_time_ms,
            total_time_ms=total_time_ms
        )

        # Step 8 — Log to SQLite
        self.logger.log(
            user_input=user_text,
            predicted_intent=response.intent,
            similarity_score=response.similarity_score,
            is_clear=response.is_clear,
            entities=response.entities,
            retrieval_time_ms=response.retrieval_time_ms,
            llm_time_ms=response.llm_time_ms,
            total_time_ms=response.total_time_ms,
            retrieved_examples=response.retrieved_examples
        )

        return response


# ── Quick Test ────────────────────────────────────────────────────────────────

async def test_pipeline():
    print("=" * 60)
    print("Testing Complete NLU Pipeline")
    print("=" * 60)

    pipeline = NLUPipeline()

    test_cases = [
        ("Book a flight to Delhi tomorrow", "book_flight"),
        ("I am hungry order me some biryani", "order_food"),
        ("Will it rain in Chennai this weekend?", "check_weather"),
        ("Cancel my hotel reservation", "cancel_booking"),
        ("Where is my food delivery?", "track_order"),
        ("Remind me to call mom at 8 PM", "set_reminder"),
        ("Play some relaxing music", "play_music"),
        ("What is the speed of light?", "out_of_scope"),
        ("", None),
    ]

    print(f"\nRunning {len(test_cases)} test cases...\n")

    passed = 0
    total = 0

    for user_input, expected_intent in test_cases:
        response = await pipeline.predict(user_input)

        if expected_intent is None:
            # Empty input test
            status = "✅" if response.intent == "unclear" else "❌"
            print(f"{status} Empty input → {response.intent}")
            if response.intent == "unclear":
                passed += 1
            total += 1
            continue

        correct = response.intent == expected_intent
        status = "✅" if correct else "❌"
        total += 1
        if correct:
            passed += 1

        print(f"{status} '{user_input[:45]}'")
        print(f"     Expected: {expected_intent}")
        print(f"     Got:      {response.intent}")
        print(f"     Entities: {response.entities}")
        print(f"     Similarity: {response.similarity_score:.3f}")
        print(f"     Latency: retrieval={response.retrieval_time_ms:.0f}ms "
              f"llm={response.llm_time_ms:.0f}ms "
              f"total={response.total_time_ms:.0f}ms")
        print()

    print("=" * 60)
    print(f"Score: {passed}/{total} correct")
    print(f"Accuracy: {passed/total*100:.1f}%")

    # Show logger stats
    stats = pipeline.logger.get_stats()
    print(f"\nLogger Stats:")
    print(f"  Total logged:  {stats['total_queries']}")
    print(f"  Avg total ms:  {stats['avg_total_ms']}")
    print(f"  Avg LLM ms:    {stats['avg_llm_ms']}")
    print(f"  Avg retrieval: {stats['avg_retrieval_ms']}")

    print("\n✅ Pipeline test complete")


if __name__ == "__main__":
    asyncio.run(test_pipeline())