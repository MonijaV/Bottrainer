import os
import time
import json
from groq import AsyncGroq
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


# ── Output Schema ─────────────────────────────────────────────────────────────

class NLUOutput(BaseModel):
    """
    Schema for LLM output.
    Groq does not support Pydantic Structured Outputs directly
    like OpenAI, so we use JSON mode + manual Pydantic parsing.
    Result is identical — clean validated object.
    """
    intent: str
    confidence: float
    entities: dict[str, str] = {}


# ── Groq Client ───────────────────────────────────────────────────────────────

class LLMClient:
    """
    Wraps Groq API for LLM inference.

    Why Groq:
    - Free tier with 14,400 requests per day
    - Faster than OpenAI (runs on custom LPU hardware)
    - Llama 3.1 8B is strong for classification tasks
    - JSON mode ensures structured output

    Why llama-3.1-8b-instant:
    - Fast response time (under 1 second typically)
    - Strong instruction following
    - Good at classification and entity extraction
    - Free on Groq tier
    """

    MODEL = "llama-3.1-8b-instant"

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_actual_groq_key_here":
            raise ValueError(
                "GROQ_API_KEY not set in .env file. "
                "Get your free key from console.groq.com"
            )
        self.client = AsyncGroq(api_key=api_key)
        print(f"LLM client initialized — Model: {self.MODEL} via Groq")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def predict(self, prompt: str) -> tuple[NLUOutput, float]:
        """
        Send prompt to Groq LLM and get structured NLU output.

        Uses Groq JSON mode to ensure valid JSON response.
        Then parses into Pydantic NLUOutput object.

        Args:
            prompt: Complete prompt built by PromptBuilder

        Returns:
            Tuple of (NLUOutput, llm_time_ms)
        """
        start = time.time()

        response = await self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise NLU classification system. "
                        "Always respond with valid JSON only. "
                        "No explanation. No markdown. Just JSON."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},  # JSON mode
            temperature=0.1,   # Low for consistent classification
            max_tokens=200     # NLU output is always short
        )

        llm_time_ms = (time.time() - start) * 1000

        # Extract raw JSON string
        raw_content = response.choices[0].message.content

        if not raw_content:
            raise ValueError("Groq returned empty response")

        # Parse JSON and validate with Pydantic
        try:
            parsed_dict = json.loads(raw_content)
            output = NLUOutput(**parsed_dict)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from Groq: {e}\nRaw: {raw_content}")
        except Exception as e:
            raise ValueError(f"Pydantic validation failed: {e}\nRaw: {raw_content}")

        return output, llm_time_ms


# ── Quick Test ────────────────────────────────────────────────────────────────

async def test_llm_client():
    print("=" * 50)
    print("Testing Groq LLM Client")
    print("=" * 50)

    client = LLMClient()

    test_prompt = """
You are a precise NLU system for a chatbot.

Available intents: book_flight, order_food, check_weather, 
cancel_booking, track_order, set_reminder, play_music, out_of_scope

Intent Definitions:
- book_flight: User wants to book or schedule air travel
- order_food: User wants to order food or drinks
- check_weather: User wants weather information
- cancel_booking: User wants to cancel a reservation
- track_order: User wants to track a delivery
- set_reminder: User wants to set a reminder or alarm
- play_music: User wants to play music
- out_of_scope: Does not match any above intent

Relevant Examples:
Example 1:
  User: "Book a flight to Delhi"
  Intent: book_flight

Example 2:
  User: "I want to fly to Mumbai tomorrow"
  Intent: book_flight

Classify this message: "Book me a flight to Delhi tomorrow"

Return ONLY this JSON format:
{
  "intent": "one of the available intents",
  "confidence": 0.0,
  "entities": {}
}
"""

    print(f"\nSending test to Groq ({client.MODEL})...")
    output, latency = await client.predict(test_prompt)

    print(f"\nResponse received in {latency:.0f}ms")
    print(f"Intent:     {output.intent}")
    print(f"Confidence: {output.confidence}")
    print(f"Entities:   {output.entities}")

    assert output.intent in [
        "book_flight", "order_food", "check_weather",
        "cancel_booking", "track_order", "set_reminder",
        "play_music", "out_of_scope"
    ], f"Unexpected intent: {output.intent}"

    print("\n✅ Groq client working correctly")
    print("✅ JSON mode returned valid structured output")
    print("✅ Pydantic validation passed")
    print(f"✅ Response time: {latency:.0f}ms")

    # Test with multiple queries
    print("\n--- Testing 5 more queries ---\n")

    test_cases = [
        ("I am hungry order me a pizza", "order_food"),
        ("Will it rain in Mumbai tomorrow?", "check_weather"),
        ("Cancel my flight reservation", "cancel_booking"),
        ("Where is my food order?", "track_order"),
        ("What is the capital of France?", "out_of_scope"),
    ]

    passed = 0
    for query, expected_intent in test_cases:
        simple_prompt = f"""
Classify this chatbot message into one of these intents:
book_flight, order_food, check_weather, cancel_booking, 
track_order, set_reminder, play_music, out_of_scope

Message: "{query}"

Return ONLY JSON: {{"intent": "intent_name", "confidence": 0.0, "entities": {{}}}}
"""
        result, ms = await client.predict(simple_prompt)
        status = "✅" if result.intent == expected_intent else "❌"
        print(f"{status} '{query}'")
        print(f"   Expected: {expected_intent} | Got: {result.intent} | {ms:.0f}ms")
        if result.intent == expected_intent:
            passed += 1

    print(f"\nScore: {passed}/5 correct")
    print("\n✅ Phase 4 LLM inference complete")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_llm_client())