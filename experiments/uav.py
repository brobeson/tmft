"""
Run UAV123 experiments.

Copyright brobeson
"""

import argparse
import got10k.experiments
import experiments.got10k_wrapper
import experiments.slack_reporter
import experiments.command_line as command_line


def main() -> None:
    """The main entry point of the UAV123 experiment runner."""
    arguments = _parse_command_line()
    experiments.got10k_wrapper.run_experiment(
        experiments.slack_reporter.make_slack_reporter(arguments.slack_file, "UAV"),
        got10k.experiments.ExperimentUAV123(
            arguments.dataset_path,
            version=arguments.version,
            result_dir=arguments.result_path,
            report_dir=arguments.report_path,
        ),
        experiments.got10k_wrapper.make_default_tracker(arguments.tracker_name),
    )


def _parse_command_line() -> argparse.Namespace:
    """
    Parse the command line.

    Returns:
        argparse.Namespace: The command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run UAV123 experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        help="The UAV dataset to use.",
        choices=["UAV123", "UAV20L"],
        default="UAV123",
    )
    command_line.add_name_option(parser)
    command_line.add_dataset_path(parser, "~/Videos/uav123")
    command_line.add_result_path(parser)
    command_line.add_report_path(parser)
    command_line.add_slack_option(parser)
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    main()
