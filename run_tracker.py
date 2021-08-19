"""
Run TMFT on a variety of datasets.

Copyright 2021 brobeson
"""

import argparse
import experiments.smoke_test


def main() -> None:
    """The main function of the application."""
    arguments = _parse_command_line()
    if arguments.smoke_test:
        experiments.smoke_test.run(arguments.smoke_test)


def _parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TMFT on a variety of datasets.")
    parser.add_argument(
        "--smoke-test",
        help="Run a smoke test on a single OTB-100 sequence. The sequence name is case sensitive.",
        metavar="SEQUENCE",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
