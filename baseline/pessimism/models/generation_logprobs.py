"""Utilities for turning generation token logprobs into likelihood metrics."""

import math
from typing import Any, Dict, List


def _safe_exp(value: float) -> float:
    """Exponentiate a log probability, allowing ordinary underflow to zero."""

    try:
        return math.exp(value)
    except OverflowError as exc:
        raise ValueError(
            f"Invalid positive log probability: {value}"
        ) from exc


def summarize_choice_logprobs(
    choice: Dict[str, Any],
) -> Dict[str, Any]:
    """Summarize chosen-token logprobs from an OpenAI chat choice.

    The OpenAI-compatible response is expected to expose
    ``choice.logprobs.content``. Only generated response tokens are included;
    prompt tokens and EOS are not part of these metrics.
    """

    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        raise ValueError(
            "Generation response does not contain choice.logprobs. "
            "Set generation_config.logprobs=true."
        )

    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError(
            "Generation response has no chosen-token logprobs in "
            "choice.logprobs.content."
        )

    token_records: List[Dict[str, Any]] = []
    token_logprobs: List[float] = []

    for token_index, token_data in enumerate(content):
        if not isinstance(token_data, dict):
            raise ValueError(
                f"Token logprob entry {token_index} is not an object."
            )

        logprob = float(token_data["logprob"])
        if not math.isfinite(logprob):
            raise ValueError(
                f"Token logprob entry {token_index} is not finite: "
                f"{logprob}"
            )

        token_logprobs.append(logprob)
        token_records.append(
            {
                "token": token_data.get("token"),
                "bytes": token_data.get("bytes"),
                "logprob": logprob,
                "probability": _safe_exp(logprob),
            }
        )

    token_count = len(token_logprobs)
    sequence_logprob = float(sum(token_logprobs))
    average_logprob = sequence_logprob / token_count

    return {
        "token_count": token_count,
        "token_logprobs": token_records,
        "sequence_logprob": sequence_logprob,
        "sequence_probability": _safe_exp(sequence_logprob),
        "sequence_nll": -sequence_logprob,
        "average_logprob": average_logprob,
        "average_nll": -average_logprob,
        "geometric_mean_token_probability": _safe_exp(
            average_logprob
        ),
    }
