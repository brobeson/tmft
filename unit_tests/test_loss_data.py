"""The the loss data module."""

import json
import os
import unittest
import utilities.loss_data


class WriteTrainingRecordsTest(unittest.TestCase):
    """Test cases for writing training records to a file."""

    def test_write_training_records(self):
        """Validate the write_training_records() function."""
        utilities.loss_data.write_training_records(
            [
                utilities.loss_data.TrainingRecord(4, [20.0, 10.0, 3.0]),
                utilities.loss_data.TrainingRecord(
                    10, [40.0, 30.0, 4.0], [200.0, 100.0, 30.0]
                ),
                utilities.loss_data.TrainingRecord(
                    14, [60.0, 50.0, 5.0], None, ["short-term"]
                ),
                utilities.loss_data.TrainingRecord(
                    20, [80.0, 70.0, 6.0], [400.0, 300.0, 40.0], ["long-term"]
                ),
            ],
            "sample_loss_data.txt",
        )
        with open("sample_loss_data.txt") as json_file:
            actual_data = json.load(json_file)
        os.remove("sample_loss_data.txt")
        self.assertEqual(
            actual_data,
            [
                {"frame_number": 4, "classification_loss": [20.0, 10.0, 3.0]},
                {
                    "frame_number": 10,
                    "classification_loss": [40.0, 30.0, 4.0],
                    "domain_adaptation_loss": [200.0, 100.0, 30.0],
                },
                {
                    "frame_number": 14,
                    "classification_loss": [60.0, 50.0, 5.0],
                    "tags": ["short-term"],
                },
                {
                    "frame_number": 20,
                    "classification_loss": [80.0, 70.0, 6.0],
                    "domain_adaptation_loss": [400.0, 300.0, 40.0],
                    "tags": ["long-term"],
                },
            ],
        )
