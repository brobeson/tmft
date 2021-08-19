"""Test the got10k_wrapper module."""

import glob
import json
import os
import shutil
import unittest
import experiments.got10k_wrapper


class WriteOtbLossDataTest(unittest.TestCase):
    """Test cases for writing OTB loss data."""

    def test_write_otb_loss_data(self):
        """Validate normal writing of OTB loss data."""
        shutil.rmtree("test_loss_data/OTBtb50/unit_test", ignore_errors=True)
        expected_loss_data = [
            [
                {
                    "frame_number": 0,
                    "classification_loss": [0, 1, 2, 3],
                    "tags": ["Basketball"],
                }
            ],
            [
                {
                    "frame_number": 0,
                    "classification_loss": [0, 1, 2, 3],
                    "tags": ["Biker"],
                }
            ],
            [
                {
                    "frame_number": 0,
                    "classification_loss": [0, 1, 2, 3],
                    "tags": ["Bird1"],
                }
            ],
            [
                {
                    "frame_number": 0,
                    "classification_loss": [0, 1, 2, 3],
                    "tags": ["BlurBody"],
                }
            ],
            [
                {
                    "frame_number": 0,
                    "classification_loss": [0, 1, 2, 3],
                    "tags": ["BlurCar2"],
                }
            ],
        ]
        # pylint: disable=protected-access
        experiments.got10k_wrapper._write_otb_loss_data(
            "./test_loss_data", "tb50", "unit_test", expected_loss_data
        )
        self.assertTrue(os.path.isdir("test_loss_data/OTBtb50/unit_test"))
        loss_files = glob.glob(
            "test_loss_data/OTBtb50/unit_test/*.txt", recursive=False
        )
        loss_files.sort()
        self.assertEqual(len(loss_files), 5)
        for i, loss_file in enumerate(loss_files):
            with self.subTest(i=i):
                with open(loss_file) as f:
                    actual_loss_data = json.load(f)
                self.assertEqual(expected_loss_data[i], actual_loss_data)
