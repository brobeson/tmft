"""
Run experiments using the GOT-10k tool.

Copyright brobeson
"""

import argparse
import datetime
import os
import sys
from typing import Union
import got10k.experiments
import got10k.trackers
import experiments.command_line as command_line
import experiments.slack_reporter as slack_reporter
import tracking.tmft

OTB_VERSIONS = ["tb50", "tb100"]
VOT_VERSIONS = [2013, 2014, 2015, 2016, 2017, 2018]
UAV_VERSIONS = ["uav123", "uav20l"]


class Got10kTmft(got10k.trackers.Tracker):
    """
    A wrapper class so the GOT-10k tool can run TMFT.

    Attributes:
        tracker (tracking.tmft.Tmft): The actual TMFT tracker.
        name (str): The tracker's name. It is used in the reports and results output.
    """

    def __init__(self, tracker: tracking.tmft.Tmft, name: str) -> None:
        super().__init__(name=name, is_deterministic=False)
        self.tracker = tracker

    def init(self, image, box):
        self.tracker.initialize(image, box)

    def update(self, image):
        return self.tracker.find_target(image)


class ConsoleReporter:
    """
    An alternative to :py:class`experiments.slack_reporter.SlackReporter`. This reporter prints
    messages to the console.
    """

    def send_message(self, message: str) -> None:  # pylint: disable=no-self-use
        """
        Print a message to the console.

        Args:
            message (str): This is not used in the method.
        """
        print(message)


def main() -> None:
    """The main entry function for running official benchmark experiments."""
    arguments = parse_command_line()
    experiment = make_experiment(arguments)
    if arguments.tracker_name is not None:
        run_tracker(experiment, arguments.tracker_name, arguments.slack_file)
    if arguments.report_trackers is not None:
        experiment.report(arguments.report_trackers)


def parse_command_line() -> argparse.Namespace:
    """
    Parse the command line.

    Returns:
        argparse.Namespace: The command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run experiments using the GOT-10k tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    command_line.add_name_option(parser)
    command_line.add_dataset_path(parser, "~/Videos")
    command_line.add_result_path(parser)
    command_line.add_report_path(parser)
    command_line.add_slack_option(parser)
    parser.add_argument(
        "--report-trackers",
        help="A list of other trackers to include in the report. This trackers must have results "
        "in the results folder.",
        nargs="+",
        metavar="TRACKER",
    )
    parser.add_argument(
        "version",
        help="The dataset to use. ",
        choices=OTB_VERSIONS + VOT_VERSIONS + UAV_VERSIONS,
    )
    arguments = parser.parse_args()
    return arguments


def make_notifier(
    configuration_file: str, source: str
) -> Union[ConsoleReporter, slack_reporter.SlackReporter]:
    """
    Make an appropriate notifier object.

    Args:
        configuration_file (str | None): The path to the Slack configuration file.
        source (str): The source to use for Slack reporting.

    Returns:
        slack_reporter.SlackReporter | ConsoleReporter: If the function can load the
        ``configuration_file``, the function returns a :py:class:`slack_reporter.SlackReporter`
        object. Otherwise, the function returns a :py:class:`ConsoleReporter` object.
    """
    if configuration_file is not None and os.path.isfile(configuration_file):
        return slack_reporter.SlackReporter(
            source, **slack_reporter.read_slack_configuration(configuration_file)
        )
    return ConsoleReporter()


def make_experiment(experiment_configuration: argparse.Namespace):
    """
    Create the GOT-10k experiment to run.

    Args:
        experiment_configuration (argparse.Namespace): The experiment configuration. This most
            likely comes from command line arguments.

    Returns:
        The GOT-10k experiment to run.
    """
    if experiment_configuration.version in OTB_VERSIONS:
        experiment = got10k.experiments.ExperimentOTB(
            experiment_configuration.dataset_path,
            experiment_configuration.version,
            report_dir=experiment_configuration.report_path,
        )
        # GOTCHA!
        # GOT-10k adds a subdirectory to the result directory in
        # ExperimentOTB.__init__(). I don't want that; it prevents reusing the tracking results
        # from tb100 for tb50.
        experiment.result_dir = experiment_configuration.result_path
        return experiment
    if experiment_configuration.version in VOT_VERSIONS:
        return got10k.experiments.ExperimentVOT(
            experiment_configuration.dataset_path,
            experiment_configuration.version,
            read_image=True,
            experiments="supervised",
            result_dir=experiment_configuration.result_path,
            report_dir=experiment_configuration.report_path,
        )
    if experiment_configuration.version in UAV_VERSIONS:
        return got10k.experiments.ExperimentUAV123(
            experiment_configuration.dataset_path,
            experiment_configuration.version.upper(),
            experiment_configuration.result_path,
            experiment_configuration.report_path,
        )
    raise ValueError(
        f"Experiment version '{experiment_configuration.version}' is unknown."
    )


def run_tracker(experiment, tracker_name: str, slack_file: str) -> None:
    """
    Run an experiment based on the GOT-10k toolkit.

    Args:
        experiment: The GOT-10k experiment object to run.
        tracker_name (str): The name of the tracker to run within the ``experiment``.
        slack_file (str | None): The Slack configuration file. If this is ``None``, console
            notifications are used.
    """
    notifier = make_notifier(slack_file, sys.platform)
    tracker = Got10kTmft(
        tracking.tmft.Tmft(
            tracking.tmft.read_configuration(
                os.path.expanduser("~/repositories/tmft/tracking/options.yaml")
            )
        ),
        name=tracker_name,
    )
    notifier.send_message(
        "Starting "
        + tracker_name
        + " experiment at "
        + datetime.datetime.today().isoformat(sep=" ", timespec="minutes")
    )
    try:
        experiment.run(tracker)
    except Exception as error:  # pylint: disable=broad-except
        notifier.send_message(f"Error during experiment: '{str(error)}'")
    else:
        notifier.send_message(
            "Experiment finished at "
            + datetime.datetime.today().isoformat(sep=" ", timespec="minutes")
        )


if __name__ == "__main__":
    main()
