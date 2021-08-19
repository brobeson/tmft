"""Collect training meta-data."""

import json

class Epoch:
    """Encapsulate training meta-data for a singe training epoch."""
    def __init__(self):
        self.learning_rates = []
        self.frame = None
        self.losses = []


def to_json(epochs):
    """
    Convert Epoch objects to JSON data.

    :param iterable epochs: An iterable container of Epoch objects.
    :returns: A list of dicts. Each dict is one Epoch object.
    """
    return [vars(e) for e in epochs]


def write_training_data(filename: str, data: Epoch) -> None:
    """
    Write training meta-data to a JSON file.

    Args:
        sequence_name (str): The name of the file to write.
        data (modules.training.Epoch): The data to write.
    """
    if isinstance(data, Epoch):
        data = [data]
    if filename is None:
        filename = "training_data.json"
    if not filename.endswith(".json"):
        filename = filename + ".json"
    with open(filename, "w") as training_file:
        json.dump(to_json(data), training_file, indent=2)
