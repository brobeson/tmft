"""
Run experiments using the GOT-10k tool.

Copyright brobeson
"""

import argparse
import datetime
import os
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


def main() -> None:
    """The main entry function for running official benchmark experiments."""
    arguments = parse_command_line()
    experiment = make_experiment(arguments)
    notifier = slack_reporter.make_slack_reporter(
        arguments.slack_file, os.uname().nodename
    )
    tracker = Got10kTmft(
        tracking.tmft.Tmft(
            tracking.tmft.read_configuration(
                os.path.expanduser("~/repositories/tmft/tracking/options.yaml")
            )
        ),
        name=arguments.tracker_name,
    )
    run_experiment(notifier, experiment, tracker)


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
        "version",
        help="The dataset to use. ",
        choices=OTB_VERSIONS + VOT_VERSIONS + UAV_VERSIONS,
    )
    arguments = parser.parse_args()
    return arguments


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
        return got10k.experiments.ExperimentOTB(
            experiment_configuration.dataset_path,
            experiment_configuration.version,
            experiment_configuration.result_path,
            experiment_configuration.report_path,
        )
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


def run_experiment(notifier, experiment, tracker) -> None:
    """
    Run an experiment based on the GOT-10k toolkit.

    Args:
        notifier: A Slack reporter to send notifications.
        experiment: The GOT-10k experiment object to run.
        tracker: The tracker to run within the ``experiment``.
    """
    notifier.send_message(
        "Starting experiment at "
        + datetime.datetime.today().isoformat(sep=" ", timespec="minutes")
    )
    experiment.run(tracker)
    experiment.report(tracker.name)
    notifier.send_message(
        "Experiment finished at "
        + datetime.datetime.today().isoformat(sep=" ", timespec="minutes")
    )


if __name__ == "__main__":
    main()
