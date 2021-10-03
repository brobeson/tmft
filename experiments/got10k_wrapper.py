"""
Encapsulate TMFT in a tracker class that GOT-10k can use.

Copyright brobeson
"""

import datetime
import os.path
import got10k.trackers
import tracking.tmft


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


def make_default_tracker(name: str) -> Got10kTmft:
    """
    Initialize a default GOT10k TMFT tracker.

    Args:
        name (str): The name of the tracker to use in results and reports.

    Returns:
        Got10kTmft: The initialized GOT10k TMFT tracker.
    """
    return Got10kTmft(
        tracking.tmft.Tmft(
            tracking.tmft.read_configuration(
                os.path.expanduser("~/repositories/tmft/tracking/options.yaml")
            )
        ),
        name=name,
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
