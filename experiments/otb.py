"""
Run OTB experiments.

Copyright brobeson
"""

import argparse
import got10k.experiments
import experiments.got10k_wrapper
import utilities.command_line as command_line


def main() -> None:
    """The main entry point of the OTB experiment runner."""
    arguments = _parse_command_line()
    print(arguments)
    experiment = got10k.experiments.ExperimentOTB(
        arguments.dataset_path,
        version=arguments.version,
        result_dir=arguments.result_path,
        report_dir=arguments.report_path,
    )
    tracker = experiments.got10k_wrapper.make_default_tracker(arguments.tracker_name)
    experiment.run(tracker)
    experiment.report(tracker)


def _parse_command_line() -> argparse.Namespace:
    """
    Parse the command line.

    Returns:
        argparse.Namespace: The command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run OTB experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        help="The OTB dataset to use.",
        choices=["tb50", "tb100"],
        default="tb100",
    )
    command_line.add_name_option(parser)
    command_line.add_dataset_path(parser, "~/Videos/otb")
    command_line.add_result_path(parser)
    command_line.add_report_path(parser)
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    main()
