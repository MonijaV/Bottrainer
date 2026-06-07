from datasets import load_dataset
import json
import os

# Map SNIPS intent names to our intent names
SNIPS_TO_OURS = {
    "GetWeather": "check_weather",
    "BookRestaurant": "order_food",
    "PlayMusic": "play_music",
    "AddToPlaylist": "play_music",
    "RateBook": "out_of_scope",
    "SearchScreeningEvent": "out_of_scope",
    "SearchCreativeWork": "out_of_scope"
}

def load_snips_eval_samples(max_per_intent: int = 20) -> list[dict]:
    """
    Load SNIPS dataset and map to our intents.
    Returns list of {text, intent} dicts.
    """
    try:
        dataset = load_dataset("snips_built_in_intents", split="train")
        samples = []
        intent_counts = {}

        for item in dataset:
            snips_intent = item.get("label")
            # Handle both string and int labels
            if isinstance(snips_intent, int):
                label_names = dataset.features["label"].names
                snips_intent = label_names[snips_intent]

            our_intent = SNIPS_TO_OURS.get(snips_intent)
            if our_intent is None:
                continue

            count = intent_counts.get(our_intent, 0)
            if count >= max_per_intent:
                continue

            samples.append({
                "text": item["text"],
                "intent": our_intent,
                "source": "snips"
            })
            intent_counts[our_intent] = count + 1

        print(f"Loaded {len(samples)} samples from SNIPS dataset")
        for intent, count in intent_counts.items():
            print(f"  {intent}: {count} samples")

        return samples

    except Exception as e:
        print(f"Could not load SNIPS dataset: {e}")
        print("Continuing without SNIPS data")
        return []


if __name__ == "__main__":
    samples = load_snips_eval_samples()
    print(f"\nTotal SNIPS samples loaded: {len(samples)}")
    if samples:
        print("\nSample entries:")
        for s in samples[:3]:
            print(f"  {s}")