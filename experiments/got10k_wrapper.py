"""
Encapsulate TMFT in a tracker class that GOT-10k can use.

Copyright brobeson
"""

import got10k.trackers
import tracking.tmft


class Got10kTmft(got10k.trackers.Tracker):
    """
    A wrapper class so the GOT-10k tool can run TMFT.

    Attributes:
        tracker (tracking.tmft.Tmft): The actual TMFT tracker.
    """

    def __init__(self, tracker: tracking.tmft.Tmft) -> None:
        super().__init__(name="TMFT", is_deterministic=False)
        self.tracker = tracker

    def init(self, image, box):
        self.tracker.initialize(image, box)

    def update(self, image):
        return self.tracker.find_target(image)
