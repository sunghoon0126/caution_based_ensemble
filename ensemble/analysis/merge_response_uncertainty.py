#!/usr/bin/env python3

import argparse
import csv
import gzip
from pathlib import Path


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "Candidate-level csv.gz files "
            "to merge."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_fields = None
    total_rows = 0

    print("=" * 80)
    print("Merging response / uncertainty files")
    print("=" * 80)

    with gzip.open(
        args.output,
        "wt",
        encoding="utf-8",
        newline="",
    ) as output_file:

        writer = None

        for input_path in args.inputs:

            if not input_path.exists():

                raise FileNotFoundError(
                    f"Input not found: "
                    f"{input_path}"
                )

            print()
            print(
                f"Reading: {input_path}"
            )

            file_rows = 0

            with gzip.open(
                input_path,
                "rt",
                encoding="utf-8",
                newline="",
            ) as input_file:

                reader = csv.DictReader(
                    input_file
                )

                fields = (
                    reader.fieldnames
                )

                if fields is None:

                    raise RuntimeError(
                        f"No header found: "
                        f"{input_path}"
                    )

                # ----------------------------------------
                # First input defines schema
                # ----------------------------------------

                if reference_fields is None:

                    reference_fields = fields

                    writer = csv.DictWriter(
                        output_file,
                        fieldnames=(
                            reference_fields
                        ),
                    )

                    writer.writeheader()

                # ----------------------------------------
                # Validate all files have same columns
                # ----------------------------------------

                elif fields != reference_fields:

                    raise RuntimeError(
                        "Column mismatch.\n"
                        f"Reference:\n"
                        f"{reference_fields}\n\n"
                        f"{input_path}:\n"
                        f"{fields}"
                    )

                # ----------------------------------------
                # Copy rows
                # ----------------------------------------

                for row in reader:

                    writer.writerow(
                        row
                    )

                    file_rows += 1
                    total_rows += 1

            print(
                f"Rows copied: "
                f"{file_rows}"
            )

    print()
    print("=" * 80)
    print("SUCCESS")
    print("=" * 80)

    print(
        f"Files merged: "
        f"{len(args.inputs)}"
    )

    print(
        f"Total candidate rows: "
        f"{total_rows}"
    )

    print(
        f"Saved: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()