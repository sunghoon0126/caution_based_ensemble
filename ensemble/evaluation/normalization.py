import json
from pathlib import Path

import numpy as np


def fit_normalization_stats(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    mean = float(
        values.mean()
    )

    std = float(
        values.std()
    )

    if std <= 0.0:
        raise ValueError(
            f"Invalid normalization std: {std}"
        )

    return {
        "mean": mean,
        "std": std,
        "num_values": int(values.size),
    }


def normalize_with_stats(
        values,
        stats,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return (
        values - stats["mean"]
    ) / stats["std"]


def save_normalization_stats(
        path,
        stats,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
        )


def load_normalization_stats(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)