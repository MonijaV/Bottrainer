import json
import os
from pydantic import BaseModel
from pathlib import Path


# ── Pydantic Models ──────────────────────────────────────────────────────────

class Intent(BaseModel):
    name: str
    examples: list[str]


class IntentsDataset(BaseModel):
    intents: list[Intent]


class EvalSample(BaseModel):
    text: str
    intent: str
    source: str = "manual"


class ExampleWithLabel(BaseModel):
    text: str
    intent: str


# ── Data Loader ──────────────────────────────────────────────────────────────

class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self._intents_dataset: IntentsDataset | None = None
        self._eval_samples: list[EvalSample] | None = None

    def load_intents(self) -> IntentsDataset:
        if self._intents_dataset is None:
            path = self.data_dir / "intents.json"
            with open(path, "r") as f:
                raw = json.load(f)
            self._intents_dataset = IntentsDataset(**raw)
            print(f"Loaded {len(self._intents_dataset.intents)} intents")
        return self._intents_dataset

    def load_eval_set(self) -> list[EvalSample]:
        if self._eval_samples is None:
            path = self.data_dir / "eval_dataset.json"
            with open(path, "r") as f:
                raw = json.load(f)
            self._eval_samples = [
                EvalSample(**s) for s in raw["eval_samples"]
            ]
            print(f"Loaded {len(self._eval_samples)} eval samples")
        return self._eval_samples

    def get_all_intent_names(self) -> list[str]:
        dataset = self.load_intents()
        return [intent.name for intent in dataset.intents]

    def get_examples_for_intent(self, intent_name: str) -> list[str]:
        dataset = self.load_intents()
        for intent in dataset.intents:
            if intent.name == intent_name:
                return intent.examples
        return []

    def get_all_examples_flat(self) -> list[ExampleWithLabel]:
        dataset = self.load_intents()
        examples = []
        for intent in dataset.intents:
            for example in intent.examples:
                examples.append(ExampleWithLabel(
                    text=example,
                    intent=intent.name
                ))
        return examples


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = DataLoader()

    intents = loader.load_intents()
    print(f"\nIntents loaded: {len(intents.intents)}")
    for intent in intents.intents:
        print(f"  {intent.name}: {len(intent.examples)} examples")

    eval_set = loader.load_eval_set()
    print(f"\nEval samples loaded: {len(eval_set)}")

    intent_names = loader.get_all_intent_names()
    print(f"\nIntent names: {intent_names}")

    all_examples = loader.get_all_examples_flat()
    print(f"\nTotal training examples: {len(all_examples)}")
    print(f"Sample: {all_examples[0]}")