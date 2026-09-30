import json
import argparse

from ensemble.evaluation.normalization import (
    fit_normalization_stats,
    save_normalization_stats,
)


ALPHAS = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
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

    # ========================================================
    # Containers
    # ========================================================

    rewards = []
    rnd_uncertainties = []

    # 1. STD
    neuboots_std = []

    # 2. Distance
    neuboots_distance = []

    # 3. Median - Quantile
    neuboots_median_quantile = {
        f"{alpha:.1f}": []
        for alpha in ALPHAS
    }

    # 4. Average - Quantile
    neuboots_avg_quantile = {
        f"{alpha:.1f}": []
        for alpha in ALPHAS
    }

    # 5. Average - Quantile's Average
    neuboots_avg_quantile_avg = {
        f"{alpha:.1f}": []
        for alpha in ALPHAS
    }

    # ========================================================
    # Read calibration scores
    # ========================================================

    with open(
        args.input_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(
                line
            )

            # ------------------------------------------------
            # Reward
            # ------------------------------------------------

            rewards.append(
                float(
                    item[
                        "reward_score"
                    ]
                )
            )

            # ------------------------------------------------
            # Original Caution RND
            # ------------------------------------------------

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

            # ------------------------------------------------
            # 1. STD
            # ------------------------------------------------

            neuboots_std.append(
                float(
                    item[
                        "neuboots_uncertainty_std"
                    ]
                )
            )

            # ------------------------------------------------
            # 2. Distance
            # ------------------------------------------------

            neuboots_distance.append(
                float(
                    item[
                        "neuboots_uncertainty_distance"
                    ]
                )
            )

            # ------------------------------------------------
            # 3-5. Quantile-based metrics
            # ------------------------------------------------

            for alpha in ALPHAS:

                alpha_key = (
                    f"{alpha:.1f}"
                )

                neuboots_median_quantile[
                    alpha_key
                ].append(
                    float(
                        item[
                            "neuboots_uncertainty_median_quantile"
                        ][
                            alpha_key
                        ]
                    )
                )

                neuboots_avg_quantile[
                    alpha_key
                ].append(
                    float(
                        item[
                            "neuboots_uncertainty_avg_quantile"
                        ][
                            alpha_key
                        ]
                    )
                )

                neuboots_avg_quantile_avg[
                    alpha_key
                ].append(
                    float(
                        item[
                            "neuboots_uncertainty_avg_quantile_avg"
                        ][
                            alpha_key
                        ]
                    )
                )

    # ========================================================
    # Basic validation
    # ========================================================

    num_examples = len(
        rewards
    )

    if num_examples == 0:
        raise RuntimeError(
            "No calibration examples found."
        )

    def check_length(
            name,
            values,
    ):
        if (
            len(values)
            != num_examples
        ):
            raise ValueError(
                f"{name}: "
                f"{len(values)} values, "
                f"expected {num_examples}"
            )

    check_length(
        "neuboots_std",
        neuboots_std,
    )

    check_length(
        "neuboots_distance",
        neuboots_distance,
    )

    for alpha in ALPHAS:

        alpha_key = (
            f"{alpha:.1f}"
        )

        check_length(
            (
                "median_quantile_"
                f"{alpha_key}"
            ),
            neuboots_median_quantile[
                alpha_key
            ],
        )

        check_length(
            (
                "avg_quantile_"
                f"{alpha_key}"
            ),
            neuboots_avg_quantile[
                alpha_key
            ],
        )

        check_length(
            (
                "avg_quantile_avg_"
                f"{alpha_key}"
            ),
            neuboots_avg_quantile_avg[
                alpha_key
            ],
        )

    # ========================================================
    # Normalization statistics
    # ========================================================

    stats = {
        "reward":
            fit_normalization_stats(
                rewards
            ),

        # 1. STD
        "neuboots_std":
            fit_normalization_stats(
                neuboots_std
            ),

        # 2. Distance
        "neuboots_distance":
            fit_normalization_stats(
                neuboots_distance
            ),
    }

    # --------------------------------------------------------
    # Keep original Caution normalization if available
    # --------------------------------------------------------

    if rnd_uncertainties:

        check_length(
            "rnd_uncertainty",
            rnd_uncertainties,
        )

        stats[
            "rnd_uncertainty"
        ] = fit_normalization_stats(
            rnd_uncertainties
        )

    # --------------------------------------------------------
    # Backward compatibility
    #
    # Old evaluator uses:
    #
    #     normalization_stats["neuboots_uncertainty"]
    #
    # It continues to mean STD.
    # --------------------------------------------------------

    stats[
        "neuboots_uncertainty"
    ] = fit_normalization_stats(
        neuboots_std
    )

    # ========================================================
    # 3-5. Alpha-dependent metrics
    # ========================================================

    for alpha in ALPHAS:

        alpha_key = (
            f"{alpha:.1f}"
        )

        # 3. Median - Quantile
        stats[
            (
                "neuboots_median_quantile_"
                f"{alpha_key}"
            )
        ] = fit_normalization_stats(
            neuboots_median_quantile[
                alpha_key
            ]
        )

        # 4. Average - Quantile
        stats[
            (
                "neuboots_avg_quantile_"
                f"{alpha_key}"
            )
        ] = fit_normalization_stats(
            neuboots_avg_quantile[
                alpha_key
            ]
        )

        # 5. Average - Quantile's Average
        stats[
            (
                "neuboots_avg_quantile_avg_"
                f"{alpha_key}"
            )
        ] = fit_normalization_stats(
            neuboots_avg_quantile_avg[
                alpha_key
            ]
        )

    # ========================================================
    # Save
    # ========================================================

    save_normalization_stats(
        args.output_path,
        stats,
    )

    # ========================================================
    # Print summary
    # ========================================================

    print()
    print(
        "Normalization statistics"
    )
    print(
        "=" * 80
    )

    for (
        name,
        value,
    ) in stats.items():

        print(
            f"{name:45s} "
            f"mean={value['mean']:.6f} "
            f"std={value['std']:.6f} "
            f"n={value['num_values']}"
        )

    print()
    print(
        f"Saved: "
        f"{args.output_path}"
    )


if __name__ == "__main__":
    main()