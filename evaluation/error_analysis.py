from collections import defaultdict


def analyze_errors(
    true_labels: list[str],
    predicted_labels: list[str],
    texts: list[str]
) -> dict:
    """
    Analyze prediction errors to find patterns.

    Groups errors by confusion pair (true → predicted).
    This is the most valuable insight from evaluation —
    shows WHERE the system fails and WHY.

    Interviewers love this because it shows engineering
    thinking beyond just accuracy numbers.
    """

    errors = []
    confusion_pairs = defaultdict(list)

    for text, true, predicted in zip(texts, true_labels, predicted_labels):
        if true != predicted:
            error = {
                "text": text,
                "true_intent": true,
                "predicted_intent": predicted
            }
            errors.append(error)

            pair_key = f"{true} → {predicted}"
            confusion_pairs[pair_key].append(text)

    # Sort confusion pairs by frequency
    sorted_pairs = sorted(
        confusion_pairs.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    top_confusions = []
    for pair, examples in sorted_pairs[:5]:
        intents = pair.split(" → ")
        top_confusions.append({
            "confusion_pair": pair,
            "true_intent": intents[0],
            "predicted_intent": intents[1],
            "count": len(examples),
            "examples": examples[:3]
        })

    return {
        "total_errors": len(errors),
        "error_rate": round(len(errors) / max(len(true_labels), 1), 4),
        "top_confusions": top_confusions,
        "all_errors": errors
    }


def generate_error_report(error_analysis: dict) -> str:
    """
    Generate human readable error analysis report.
    This goes in your README and GitHub documentation.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("ERROR ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append(f"Total errors: {error_analysis['total_errors']}")
    lines.append(f"Error rate:   {error_analysis['error_rate']*100:.1f}%")
    lines.append("")

    if not error_analysis["top_confusions"]:
        lines.append("No errors found — perfect classification!")
        return "\n".join(lines)

    lines.append("TOP CONFUSION PAIRS")
    lines.append("-" * 60)

    for confusion in error_analysis["top_confusions"]:
        lines.append(f"\n{confusion['confusion_pair']} "
                     f"({confusion['count']} times)")
        lines.append("  Examples that caused confusion:")
        for example in confusion["examples"]:
            lines.append(f"    - \"{example}\"")

    lines.append("")
    lines.append("=" * 60)
    lines.append("SUGGESTED FIXES")
    lines.append("-" * 60)
    lines.append("For each confusion pair above:")
    lines.append("1. Add more contrastive examples to intents.json")
    lines.append("2. Make intent definitions more specific in prompt")
    lines.append("3. Re-run evaluation to measure improvement")

    return "\n".join(lines)