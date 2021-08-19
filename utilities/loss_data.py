"""
utilities.loss_data
===================

Functionality for recording training loss data.
"""

import json
import os.path


class TrainingRecord:
    """
    The loss data from one round of training.
    """

    def __init__(
        self,
        frame_number: int,
        classification_loss,
        domain_adaptation_loss=None,
        tags=None,
    ):
        self.frame_number = frame_number
        self.classification_loss = classification_loss
        if domain_adaptation_loss is not None:
            self.domain_adaptation_loss = domain_adaptation_loss
        if tags is not None:
            self.tags = tags


class _TrainingRecordEncoder(json.JSONEncoder):
    """A custom JSON encoder for TrainingRecord instances."""

    def default(self, o):  # pylint: disable=method-hidden
        """Convert the TrainingRecord ``o`` into a type that can be serialized as JSON."""
        return o.__dict__


def write_training_records(records, path: str) -> None:
    """
    Write a set of training records to a JSON file.

    :param records: The training record, or records, to write.
    :param str path: The path to the file to write.
    """
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, "w") as json_file:
        json.dump(records, json_file, indent=2, cls=_TrainingRecordEncoder)
