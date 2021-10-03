"""
Run VOT experiments.

Copyright brobeson
"""

import argparse
import got10k.experiments
import experiments.got10k_wrapper
import utilities.command_line as command_line


def main() -> None:
    """The main entry point of the VOT experiment runner."""
    arguments = _parse_command_line()
    print(arguments)
    experiment = got10k.experiments.ExperimentVOT(
        arguments.dataset_path,
        version=arguments.version,
        # TODO What do the different experiment types mean?
        experiments="supervised",
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
        description="Run VOT experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        help="The VOT dataset to use.",
        choices=[2013, 2014, 2015, 2016, 2017, 2018],
        default="2018",
    )
    command_line.add_name_option(parser)
    command_line.add_dataset_path(parser, "~/Videos/vot-got")
    command_line.add_result_path(parser)
    command_line.add_report_path(parser)
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    main()