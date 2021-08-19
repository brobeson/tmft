"""
networks.domain_adaptation_schedules
====================================

Different learning rate schedules for the domain adaptation layers.
"""

import math
import numpy
import torch.autograd
import torch.nn


class GradientReverseLayer(torch.autograd.Function):
    """Provides a gradient reverse layer with scheduling capability."""

    schedule = None

    @staticmethod
    def forward(ctx, input_tensor):  # pylint: disable=arguments-differ
        ctx.save_for_backward(input_tensor)
        return input_tensor
        # return GradientReverseLayer.schedule.forward(input_tensor)

    @staticmethod
    def backward(ctx, output_gradients):  # pylint: disable=arguments-differ
        return -GradientReverseLayer.schedule.learning_rate() * output_gradients

    # @staticmethod
    # def reset():
    #     """
    #     Reset the gradient reverse layer and the learning rate schedule.

    #     Each schedule must implement a reset() method. The primary use case
    #     is to reset the schedule to iteration 0, but the schedule can do whatever it needs to do.
    #     """
    #     GradientReverseLayer.schedule.reset()


grl = GradientReverseLayer.apply  # pylint: disable=invalid-name


class Schedule:
    """Interface for TMFT learning rate schedules."""

    def __init__(self, minimum_rate: float, maximum_rate: float, batches: int):
        self._minimum_rate = minimum_rate
        self._maximum_rate = maximum_rate
        self._batches = batches if batches is None else float(batches)
        self._current_batch = 0

    # def forward(self, x):
    #     """Feed data forward through the layer."""
    #     return x

    def learning_rate(self):
        """
        Get the schedule's learning rate for the current batch.

        Subclasses must implement this method.
        """
        raise NotImplementedError(
            "Schedule.learning_rate() must be defined by a derived class."
        )

    def reset(self) -> None:
        """Reset the schedule for a new training epoch."""
        self._current_batch = 0

    def step(self) -> None:
        """Step the schedule to the next batch."""
        self._current_batch += 1


class DecreasingPada(Schedule):
    """Implements the PADA schedule with a decreasing trend."""

    def __init__(
        self, minimum_rate, maximum_rate, batches: float, lambda_: float, alpha: float,
    ):
        super().__init__(minimum_rate, maximum_rate, batches)
        self.__lambda = lambda_
        self.__alpha = alpha

    def learning_rate(self):
        y = self.__lambda * (
            2.0
            * (self._maximum_rate - self._minimum_rate)
            / (1.0 + numpy.exp(self.__alpha * self._current_batch / self._batches))
            + self._minimum_rate
        )
        return y


class IncreasingPada(Schedule):
    """Implements the PADA schedule with an increasing trend."""

    def __init__(
        self, minimum_rate, maximum_rate, batches: int, lambda_: float, alpha: float,
    ):
        super().__init__(minimum_rate, maximum_rate, batches)
        self.__lambda = lambda_
        self.__alpha = alpha

    def learning_rate(self):
        y = self.__lambda * (
            2.0
            * (self._maximum_rate - self._minimum_rate)
            / (1.0 + numpy.exp(-self.__alpha * self._current_batch / self._batches))
            - (self._maximum_rate - self._minimum_rate)
            + self._minimum_rate
        )
        return y


class DecreasingCosineAnnealing(Schedule):
    """Implements the cosine annealing gradient reverse layer."""

    def __init__(self, minimum_rate, maximum_rate, batches: int):
        """
        Initialize a cosine annealing function.

        :param numpy.ndarray learning_rate_bounds: The minimum and maximum learning rate values.
        :type learning_rate_bounds: numpy.ndarray
        :param int batches: The maximum number of training batches.
        """
        super().__init__(minimum_rate, maximum_rate, batches)

    def learning_rate(self):
        """Calculate the back propagation for the layer."""
        return self._minimum_rate + 0.5 * (self._maximum_rate - self._minimum_rate) * (
            1 + math.cos(math.pi * self._current_batch / self._batches)
        )


class IncreasingCosineAnnealing(Schedule):
    """Implements the inverse cosine annealing gradient reverse layer."""

    # def __init__(self, minimum_rate: float, maximum_rate: float, batches: int):
    #     """Initialize an inverse cosine annealing function."""
    #     super().__init__(minimum_rate, maximum_rate, batches)

    def learning_rate(self):
        """Calculate the new learning rate."""
        return self._maximum_rate + 0.5 * (self._minimum_rate - self._maximum_rate) * (
            1 + math.cos(math.pi * self._current_batch / self._batches)
        )


class IncreasingGamma(Schedule):
    """Implements the increasing gamma gradient schedule."""

    def __init__(self, minimum_rate: float, maximum_rate, gamma: float, batches: int):
        super().__init__(minimum_rate, maximum_rate, batches)
        self.__gamma = gamma

    def learning_rate(self):
        return (
            math.pow(self._current_batch / self._batches, 1.0 / self.__gamma)
            * (self._maximum_rate - self._minimum_rate)
            + self._minimum_rate
        )


class DecreasingGamma(Schedule):
    """Implements the gamma gradient reverse layer."""

    def __init__(
        self, minimum_rate: float, maximum_rate: float, gamma: float, batches: int
    ):
        super().__init__(minimum_rate, maximum_rate, batches)
        self.__gamma = gamma

    def learning_rate(self):
        return (
            -math.pow(self._current_batch / self._batches, 1.0 / self.__gamma)
            * (self._maximum_rate - self._minimum_rate)
            + self._maximum_rate
        )


class IncreasingLinear(Schedule):
    """Implements an increasing linear learning rate schedule."""

    # def __init__(self, minimum_rate: float, maximum_rate: float, batches: int):
    #     super().__init__(minimum_rate, maximum_rate, batches)

    def learning_rate(self):
        """Calculate the new learning rate."""
        return (self._current_batch / self._batches) * (
            self._maximum_rate - self._minimum_rate
        ) + self._minimum_rate


class DecreasingLinear(Schedule):
    """Implements a decreasing linear learning rate schedule."""

    # def __init__(self, minimum_rate: float, maximum_rate: float, batches: int):
    #     super().__init__(minimum_rate, maximum_rate, batches)

    def learning_rate(self):
        """Calculate the new learning rate."""
        return (-self._current_batch / self._batches + 1.0) * (
            self._maximum_rate - self._minimum_rate
        ) + self._minimum_rate


class IncreasingExponential(Schedule):
    """Implements an increasing exponential learning rate schedule."""

    def __init__(self, maximum_rate: float, gamma: float) -> None:
        super().__init__(0.0, maximum_rate, None)
        self.__gamma = gamma

    def learning_rate(self):
        """Calculate the new learning rate."""
        return (
            -self._maximum_rate * self.__gamma ** self._current_batch
            + self._maximum_rate
        )


class DecreasingExponential(Schedule):
    """Implements a decreasing exponential learning rate schedule."""

    def __init__(self, maximum_rate: float, gamma: float) -> None:
        super().__init__(0.0, maximum_rate, None)
        self.__gamma = gamma

    def learning_rate(self):
        """Calculate the new learning rate."""
        return self._maximum_rate * self.__gamma ** self._current_batch


class IncreasingArccosine(Schedule):
    """Implements an increasing arccosine learning rate schedule."""

    # def __init__(self, minimum_rate: float, maximum_rate: float, batches: int):
    #     super().__init__(minimum_rate, maximum_rate, batches)

    def learning_rate(self):
        """Calculate the new learning rate."""
        return (
            -math.acos(2.0 * self._current_batch / self._batches - 1.0) / math.pi + 1.0
        ) * (self._maximum_rate - self._minimum_rate) + self._minimum_rate


class DecreasingArccosine(Schedule):
    """Implements a decreasing arccosine learning rate schedule."""

    # def __init__(self, minimum_rate: float, maximum_rate: float, batches: int):
    #     super().__init__(minimum_rate, maximum_rate, batches)

    def learning_rate(self):
        """Calculate the new learning rate."""
        return (
            math.acos((2.0 * self._current_batch) / self._batches - 1.0)
            / math.pi
            * (self._maximum_rate - self._minimum_rate)
            + self._minimum_rate
        )


class Constant(Schedule):
    """Implements a constant learning rate schedule."""

    def __init__(self, constant):
        super().__init__(None, None, None)
        self.__constant = constant

    def learning_rate(self):
        """Calculate the new learning rate."""
        return self.__constant


def set_schedule(schedule_type: str, direction: str, **parameters):
    """Set the schedule for the gradient reverse layer."""
    if schedule_type == "constant":
        GradientReverseLayer.schedule = Constant(**parameters)
        return
    elif not _is_direction_valid(direction):
        raise ValueError(direction + " is not a valid schedule direction.")
    if direction == "decreasing":
        _set_decreasing_schedule(schedule_type, parameters)
    _set_increasing_schedule(schedule_type, parameters)


def _set_decreasing_schedule(schedule_type: str, parameters):
    if schedule_type == "pada":
        GradientReverseLayer.schedule = DecreasingPada(**parameters)
    elif schedule_type == "linear":
        GradientReverseLayer.schedule = DecreasingLinear(**parameters)
    elif schedule_type == "cosine_annealing":
        GradientReverseLayer.schedule = DecreasingCosineAnnealing(**parameters)
    elif schedule_type == "gamma":
        GradientReverseLayer.schedule = DecreasingGamma(**parameters)
    elif schedule_type == "exponential":
        GradientReverseLayer.schedule = DecreasingExponential(**parameters)
    elif schedule_type == "arccosine":
        GradientReverseLayer.schedule = DecreasingArccosine(**parameters)
    else:
        raise ValueError("Unknown gradient reverse schedule: " + schedule_type)


def _set_increasing_schedule(schedule_type: str, parameters):
    if schedule_type == "pada":
        GradientReverseLayer.schedule = IncreasingPada(**parameters)
    elif schedule_type == "linear":
        GradientReverseLayer.schedule = IncreasingLinear(**parameters)
    elif schedule_type == "cosine_annealing":
        GradientReverseLayer.schedule = IncreasingCosineAnnealing(**parameters)
    elif schedule_type == "gamma":
        GradientReverseLayer.schedule = IncreasingGamma(**parameters)
    elif schedule_type == "exponential":
        GradientReverseLayer.schedule = IncreasingExponential(**parameters)
    elif schedule_type == "arccosine":
        GradientReverseLayer.schedule = IncreasingArccosine(**parameters)
    else:
        raise ValueError("Unknown gradient reverse schedule: " + schedule_type)


def _is_direction_valid(direction: str) -> bool:
    """
    Return if a direction string from the configuration is valid.

    :param str direction: The direction string.
    :return: True if the direction is valid, false if it is not.
    :rtype: bool
    """
    return direction.lower() in ["decreasing", "increasing"]
