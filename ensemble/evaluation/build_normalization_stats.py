import json
import argparse

import numpy as np

from ensemble.evaluation.normalization import (
    fit_normalization_stats,
    save_normalization_stats,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    rewards = []
    rnd_uncertainties = []
    neuboots_uncertainties = []

    with open(
        args.input_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            rewards.append(
                float(
                    item["reward_score"]
                )
            )

            if (
                "rnd_uncertainty"
                in item
            ):
                rnd_uncertainties.append(
                    float(
                        item[
                            "rnd_uncertainty"
                        ]
                    )
                )

            if (
                "neuboots_uncertainty"
                in item
            ):
                neuboots_uncertainties.append(
                    float(
                        item[
                            "neuboots_uncertainty"
                        ]
                    )
                )

    stats = {
        "reward": (
            fit_normalization_stats(
                rewards
            )
        )
    }

    if rnd_uncertainties:
        stats[
            "rnd_uncertainty"
        ] = fit_normalization_stats(
            rnd_uncertainties
        )

    if neuboots_uncertainties:
        stats[
            "neuboots_uncertainty"
        ] = fit_normalization_stats(
            neuboots_uncertainties
        )

    save_normalization_stats(
        args.output_path,
        stats,
    )

    print()
    print(
        "Normalization statistics"
    )
    print(
        "=" * 60
    )

    for name, value in stats.items():
        print(
            f"{name:25s} "
            f"mean={value['mean']:.6f} "
            f"std={value['std']:.6f} "
            f"n={value['num_values']}"
        )

    print()
    print(
        f"Saved: {args.output_path}"
    )


if __name__ == "__main__":
    main()