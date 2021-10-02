"""
Encapsulate TMFT in a tracker class that GOT-10k can use.

Copyright brobeson
"""

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
