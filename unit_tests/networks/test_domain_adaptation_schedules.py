"""Test the CosineAnnealing class."""

import collections
import unittest
import networks.domain_adaptation_schedules


Row = collections.namedtuple("Row", ["schedule", "learning_rate_results"])


class SchedulesTest(unittest.TestCase):
    """Test cases for the domain adaptation learning rate schedules."""

    def setUp(self):
        """
        Create the data table used for data driven testing.

        All the test methods use the same data table, so this can be done in the setUp() method.
        """
        self.test_data = [
            Row(
                networks.domain_adaptation_schedules.DecreasingCosineAnnealing(
                    0.2, 0.8, 200
                ),
                [0.8, 0.5, 0.2],
            ),
            Row(
                networks.domain_adaptation_schedules.IncreasingCosineAnnealing(
                    0.2, 0.8, 200
                ),
                [0.2, 0.5, 0.8],
            ),
            Row(
                networks.domain_adaptation_schedules.IncreasingPada(
                    0.2, 0.8, 200, 1.0, 10.0
                ),
                [0.2, 0.79197, 0.79995],
            ),
            Row(
                networks.domain_adaptation_schedules.DecreasingPada(
                    0.2, 0.8, 200, 1.0, 10.0
                ),
                [0.8, 0.20803, 0.200054],
            ),
            Row(
                networks.domain_adaptation_schedules.IncreasingGamma(
                    0.2, 0.8, 0.15, 200
                ),
                [0.2, 0.20591, 0.8],
            ),
            Row(
                networks.domain_adaptation_schedules.DecreasingGamma(
                    0.2, 0.8, 0.15, 200
                ),
                [0.8, 0.79409, 0.2],
            ),
            Row(
                networks.domain_adaptation_schedules.IncreasingLinear(0.2, 0.8, 200.0),
                [0.2, 0.5, 0.8],
            ),
            Row(
                networks.domain_adaptation_schedules.DecreasingLinear(0.2, 0.8, 200.0),
                [0.8, 0.5, 0.2],
            ),
            Row(
                networks.domain_adaptation_schedules.Constant(0.75), [0.75, 0.75, 0.75],
            ),
            Row(
                networks.domain_adaptation_schedules.DecreasingArccosine(0.2, 0.8, 200),
                [0.8, 0.5, 0.2],
            ),
            Row(
                networks.domain_adaptation_schedules.IncreasingArccosine(0.2, 0.8, 200),
                [0.2, 0.5, 0.8],
            ),
            Row(
                networks.domain_adaptation_schedules.DecreasingExponential(0.8, 0.965),
                [0.8, 0.022689, 0.00064],
            ),
            Row(
                networks.domain_adaptation_schedules.IncreasingExponential(0.8, 0.965),
                [0.0, 0.77731, 0.799356],
            ),
        ]

    # def test_forward(self):
    #     """Validate the forward() method of each schedule."""
    #     for test in self.test_data:
    #         with self.subTest(schedule=type(test.schedule).__name__):
    #             self.assertEqual(test.schedule.forward(1.0), 1.0)
    #             self.assertEqual(test.schedule.forward(2.345), 2.345)

    def test_learning_rate(self):
        """Validate the learning_rate() method of each schedule."""
        for test in self.test_data:
            with self.subTest(schedule=type(test.schedule).__name__):
                self.assertAlmostEqual(
                    test.schedule.learning_rate(),
                    test.learning_rate_results[0],
                    places=5,
                )
                for _ in range(0, 100):
                    test.schedule.step()
                self.assertAlmostEqual(
                    test.schedule.learning_rate(),
                    test.learning_rate_results[1],
                    places=5,
                )
                for _ in range(0, 100):
                    test.schedule.step()
                self.assertAlmostEqual(
                    test.schedule.learning_rate(),
                    test.learning_rate_results[2],
                    places=5,
                )
