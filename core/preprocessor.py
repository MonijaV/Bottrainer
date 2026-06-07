import re


def preprocess(text: str) -> str:
    """
    Clean user input before sending to pipeline.

    Steps:
    1. Strip leading and trailing whitespace
    2. Collapse multiple spaces into one
    3. Truncate very long inputs
    4. Handle None input safely

    Why preprocessing matters:
    Raw user input is messy. Extra spaces, weird characters,
    and very long inputs can affect embedding quality.
    Clean input = better retrieval = better classification.
    """

    if not text:
        return ""

    # Strip whitespace
    text = text.strip()

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Truncate if too long — embeddings work best under 512 tokens
    # 500 characters is a safe practical limit for NLU queries
    if len(text) > 500:
        text = text[:500]
        print(f"Warning: Input truncated to 500 characters")

    return text


def is_valid_input(text: str) -> bool:
    """
    Check if input is worth processing.
    Returns False for empty or whitespace only input.
    """
    return bool(text and text.strip())


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        ("  Book a flight to Delhi  ", "Book a flight to Delhi"),
        ("order   me   a   pizza", "order me a pizza"),
        ("", ""),
        ("   ", ""),
        ("a" * 600, "a" * 500),
    ]

    print("Testing preprocessor...\n")
    all_passed = True

    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = preprocess(input_text)
        passed = result == expected
        status = "✅" if passed else "❌"
        print(f"{status} Test {i}: '{input_text[:30]}...' → '{result[:30]}'")
        if not passed:
            all_passed = False
            print(f"   Expected: '{expected[:30]}'")
            print(f"   Got:      '{result[:30]}'")

    print(f"\n{'✅ All preprocessor tests passed' if all_passed else '❌ Some tests failed'}")