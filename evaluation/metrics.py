import json
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import pandas as pd


def calculate_metrics(
    true_labels: list[str],
    predicted_labels: list[str],
    intent_names: list[str]
) -> dict:
    """
    Calculate complete evaluation metrics.

    Returns accuracy, per-intent precision/recall/F1,
    and overall weighted averages.
    """

    accuracy = accuracy_score(true_labels, predicted_labels)

    # Per intent metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=intent_names,
        average=None,
        zero_division=0
    )

    # Weighted averages
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0
    )

    # Per intent breakdown
    per_intent = {}
    for i, intent in enumerate(intent_names):
        per_intent[intent] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i])
        }

    return {
        "accuracy": round(float(accuracy), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_intent": per_intent,
        "total_samples": len(true_labels),
        "correct": sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)
    }


def generate_classification_report(
    true_labels: list[str],
    predicted_labels: list[str],
    intent_names: list[str]
) -> str:
    """Generate sklearn classification report as string."""
    return classification_report(
        true_labels,
        predicted_labels,
        labels=intent_names,
        zero_division=0
    )


def get_confusion_matrix(
    true_labels: list[str],
    predicted_labels: list[str],
    intent_names: list[str]
) -> list[list[int]]:
    """Return confusion matrix as 2D list."""
    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=intent_names
    )
    return cm.tolist()