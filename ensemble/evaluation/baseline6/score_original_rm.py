"""Reuse the existing scalar RM scorer (do not use --compact: BP needs text)."""

from ensemble.evaluation.baseline6.common import rows
from ensemble.evaluation.score_reference_candidates import parse_args, run


def main():
    args = parse_args()
    if args.compact:
        raise ValueError("Baseline 6 requires prompt/response; do not use --compact")
    if args.output_path.exists() and not args.overwrite:
        # The shared scorer verifies IDs, response hashes and settings. Also
        # verify prompt and every source field before reusing prior full rows.
        source = iter(rows(args.input_path))
        for completed in rows(args.output_path):
            original = next(source, None)
            if original is None or any(completed.get(k) != v for k, v in original.items()):
                raise ValueError("Original RM resume source fields differ")
    run(args)

if __name__ == "__main__":
    main()
