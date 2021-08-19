# pylint: disable=arguments-differ,no-member,not-callable
import unittest
import torch
import networks.domain_adaptation_schedules


class Network(torch.nn.Module):
    """A simple neural network for testing the GRL."""

    def __init__(self, schedule: str, **schedule_parameters):
        super().__init__()
        self.hidden_layer = torch.nn.Linear(9, 6, bias=False)
        self.hidden_layer.weight.data.fill_(0.5)
        self.sigmoid = torch.nn.Sigmoid()
        self.loss_function = torch.nn.BCELoss()
        networks.domain_adaptation_schedules.set_schedule(
            schedule, "increasing", **schedule_parameters
        )

    def forward(self, x):
        # return self.sigmoid(self.hidden_layer(x))
        return networks.domain_adaptation_schedules.grl(
            self.sigmoid(self.hidden_layer(x))
        )


def generate_test_data(iterations: int, schedule: str, **schedule_parameters):
    """
    Do the heavy lifting for each test case.

    :param int iterations: The number of training iterations to run.
    :param str schedule: The type of learning rate schedule to use for the gradient reverse layer.
    :param schedule_parameters: The parameters to forward on to the schedule constructor.
    :return: The features after all training iterations have run.
    """
    network = Network(schedule, **schedule_parameters)
    network.train()
    for _ in range(iterations):
        features = torch.tensor(
            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0], requires_grad=True
        )
        output = network(features)
        network.zero_grad()
        network.loss_function(output, torch.ones_like(output)).backward()
    return features


class GradientReverseLayerTest(unittest.TestCase):
    """Test cases for the gradient reverse layer with different schedules."""

    def test_unknown_schedule(self):
        """Validate set_schedule() when the schedule is invalid."""
        with self.assertRaises(ValueError):
            networks.domain_adaptation_schedules.set_schedule("imaginary", "decreasing")

    def test_invalid_direction(self):
        with self.assertRaises(ValueError):
            networks.domain_adaptation_schedules.set_schedule(
                "linear", "increasin", minimum_rate=0.0, maximum_rate=1.0
            )

    def test_constant_schedule(self):
        """Test the Constant learning rate schedule."""
        features = generate_test_data(1, "constant", constant=1.0)
        self.assertTrue(
            torch.allclose(features.grad, torch.full_like(features, 6.1691e-5), 1.0e-7)
        )
        features = generate_test_data(100, "constant", constant=1.0)
        self.assertTrue(
            torch.allclose(features.grad, torch.full_like(features, 6.1691e-5), 1.0e-7)
        )

    def test_cosine_annealing_schedule(self):
        """Test the Cosine Annealing schedule."""
        expected_gradients = [
            torch.full((9,), 4.93528e-5),
            torch.full((9,), 3.0845e-5),
            torch.full((9,), 1.23382e-5),
        ]
        for i, expected in zip([1, 50, 100], expected_gradients):
            with self.subTest(iterations=i):
                features = generate_test_data(
                    i,
                    "cosine_annealing",
                    minimum_rate=0.2,
                    maximum_rate=0.8,
                    epochs=100,
                )
                self.assertTrue(torch.allclose(features.grad, expected, 1.0e-7))

    def test_gamma_schedule(self):
        """Test the Gamma schedule."""
        expected_gradients = [
            torch.full((9,), 1.23382e-5),
            torch.full((9,), 1.2703e-5),
            torch.full((9,), 4.93528e-5),
        ]
        for i, expected in zip([1, 50, 100], expected_gradients):
            with self.subTest(iterations=i):
                features = generate_test_data(
                    i,
                    "gamma",
                    minimum_rate=0.2,
                    maximum_rate=0.8,
                    gamma=0.15,
                    epochs=100,
                )
                self.assertTrue(torch.allclose(features.grad, expected, 1.0e-7))

    def test_inverse_cosine_annealing_schedule(self):
        """Test the Inverse Cosine Annealing schedule."""
        expected_gradients = [
            torch.full((9,), 1.23382e-5),
            torch.full((9,), 3.0845e-5),
            torch.full((9,), 4.93528e-5),
        ]
        for i, expected in zip([1, 50, 100], expected_gradients):
            with self.subTest(iterations=i):
                features = generate_test_data(
                    i,
                    "inverse_cosine_annealing",
                    minimum_rate=0.2,
                    maximum_rate=0.8,
                    epochs=100,
                )
                self.assertTrue(torch.allclose(features.grad, expected, 1.0e-7))

    def test_linear_schedule(self):
        """Test the Linear schedule."""
        expected_gradients = [
            torch.full((9,), 1.2708e-5),
            torch.full((9,), 3.0845e-5),
            torch.full((9,), 4.93528e-5),
        ]
        for i, expected in zip([1, 50, 100], expected_gradients):
            with self.subTest(iterations=i):
                features = generate_test_data(
                    i, "linear", minimum_rate=0.2, maximum_rate=0.8, epochs=100,
                )
                self.assertTrue(torch.allclose(features.grad, expected, 1.0e-7))

    def test_pada_schedule(self):
        """Test the PADA schedule."""
        expected_gradients = [
            torch.full((9,), 1.4187e-5),
            torch.full((9,), 4.8857e-5),
            torch.full((9,), 4.93528e-5),
        ]
        for i, expected in zip([1, 50, 100], expected_gradients):
            with self.subTest(iterations=i):
                features = generate_test_data(
                    i,
                    "pada",
                    minimum_rate=0.2,
                    maximum_rate=0.8,
                    lambda_=1.0,
                    alpha=10.0,
                    epochs=100,
                )
                self.assertTrue(torch.allclose(features.grad, expected, 1.0e-7))
