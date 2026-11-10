"""Strict final-answer parsing and source-specific multiple-choice normalization."""

import re

CHOICE_INSTRUCTION = (
    "Reason through the question step by step. Choose exactly one of the listed "
    "options. End with a separate line containing only #### followed by the "
    "option letter, for example: #### B"
)


def extract_choice_answer(response, labels):
    # Require the final nonempty line; never search reasoning for a bare letter.
    if not isinstance(response, str) or not response.strip():
        return None
    match = re.fullmatch(r"####\s+([A-Z])", response.strip().splitlines()[-1].strip())
    return match[1] if match and match[1] in labels else None


def normalize_choice_row(dataset, row):
    if dataset == "logiqa2":
        options = row["options"]
        index = row["answer"]
        if type(index) is not int:
            raise ValueError("LogiQA answer must be a zero-based integer")
        question = f"Context:\n{row['text']}\n\nQuestion:\n{row['question']}"
        source_labels = list(range(len(options)))
    elif dataset == "arc_challenge":
        options = row["choices"]["text"]
        source_labels = [str(x) for x in row["choices"]["label"]]
        if len(source_labels) != len(options) or len(set(source_labels)) != len(options):
            raise ValueError("ARC choices require unique labels matching their texts")
        try:
            index = source_labels.index(str(row["answerKey"]))
        except ValueError as error:
            raise ValueError("ARC answerKey is not a listed choice") from error
        question = row["question"]
    else:
        raise ValueError(f"Unknown multiple-choice dataset: {dataset}")
    if not 2 <= len(options) <= 26 or not all(isinstance(x, str) and x.strip() for x in options):
        raise ValueError("Expected 2–26 nonempty choice texts")
    if not 0 <= index < len(options):
        raise ValueError("Answer index is outside the choices")
    labels = [chr(65 + i) for i in range(len(options))]
    text = question + "\n\nOptions:\n" + "\n".join(
        f"{label}. {option}" for label, option in zip(labels, options)
    )
    return text, labels[index], {
        "source_id": row.get("id"),
        "choice_labels": labels,
        "choice_texts": list(options),
        "source_choice_labels": source_labels,
    }
