from jinja2 import Environment, FileSystemLoader, Template
from pathlib import Path


class PromptBuilder:
    """
    Builds dynamic prompts by combining:
    - Task instructions (from Jinja2 template)
    - Retrieved similar examples (from FAISS)
    - Available intent definitions
    - User query

    Why Jinja2: Clean separation of prompt logic from Python code.
    Prompts can be updated without touching Python files.
    """

    TEMPLATE_FILE = "intent_template.j2"

    def __init__(self, templates_dir: str = "prompts"):
        self.templates_dir = Path(templates_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.template = self.env.get_template(self.TEMPLATE_FILE)
        print(f"Prompt template loaded from: {self.templates_dir / self.TEMPLATE_FILE}")

    def build(self,
              user_query: str,
              retrieved_examples: list,
              valid_intents: list[str]) -> str:
        """
        Build a complete prompt for LLM inference.

        Args:
            user_query: The user's input text
            retrieved_examples: List of RetrievalResult from FAISS
            valid_intents: List of valid intent names

        Returns:
            Complete prompt string ready to send to LLM
        """
        if not user_query.strip():
            raise ValueError("User query cannot be empty")

        if not retrieved_examples:
            raise ValueError("Must provide at least one retrieved example")

        prompt = self.template.render(
            query=user_query,
            examples=retrieved_examples,
            intents=valid_intents
        )

        return prompt

    def get_prompt_stats(self, prompt: str) -> dict:
        """
        Return basic stats about a built prompt.
        Useful for debugging and monitoring.
        """
        lines = prompt.strip().split('\n')
        words = prompt.split()
        # Rough token estimate (1 token ≈ 4 characters)
        estimated_tokens = len(prompt) // 4

        return {
            "lines": len(lines),
            "words": len(words),
            "characters": len(prompt),
            "estimated_tokens": estimated_tokens
        }


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from retrieval.embeddings import EmbeddingModel
    from retrieval.faiss_retriever import FAISSRetriever

    print("=" * 60)
    print("Testing Prompt Builder")
    print("=" * 60)

    # Load retriever
    embedding_model = EmbeddingModel()
    retriever = FAISSRetriever(embedding_model)
    retriever.load(
        index_path="retrieval/faiss_index/index.faiss",
        metadata_path="retrieval/faiss_index/metadata.json"
    )

    # Build prompt builder
    builder = PromptBuilder()

    valid_intents = [
        "book_flight", "order_food", "check_weather",
        "cancel_booking", "track_order", "set_reminder",
        "play_music", "out_of_scope"
    ]

    # Test with different queries
    test_queries = [
        "Arrange air travel to Hyderabad this weekend",
        "I am hungry order me something",
        "who is the prime minister?"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 60)

        # Get retrieved examples
        examples = retriever.search(query, k=3)

        # Build prompt
        prompt = builder.build(query, examples, valid_intents)
        stats = builder.get_prompt_stats(prompt)

        print(prompt)
        print("-" * 60)
        print(f"Stats: {stats}")
        print()

        user_input = input("Press Enter for next query (or q to quit): ")
        if user_input.lower() == 'q':
            break

    print("\n✅ Prompt builder working correctly")