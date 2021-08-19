"""
utilities.sample_generator
==========================

Classes and functions to generate samples from an image.
"""

from typing import Iterable
import numpy
import utilities.bounding_boxes


class Generator:
    """
    Base class for generating samples.

    This class cannot be used directly. A derived class must be used instead.

    .. py:attribute:: image_size
    .. py:attribute:: aspect_ratio
    """

    def __init__(
        self,
        image_size: numpy.array,
        translation: float = 1.0,
        scale: float = 1.0,
        aspect_ratio=None,
        valid=False,
    ):
        self.image_size = image_size
        self.translation = translation
        self.scale = scale
        self.aspect_ratio = aspect_ratio
        self.valid = valid  # TODO What is this? Is it used?

    def generate_samples(
        self, bounding_box, n: int, overlap_range=None, scale_range=None
    ):
        """
        Generate random samples around a bounding box.

        :param bounding_box: [description]
        :type bounding_box: [type]
        :param n: [description]
        :type n: int
        :raises NotImplementedError: [description]
        """
        centered_box = utilities.bounding_boxes.centered_box(bounding_box)
        if overlap_range is None and scale_range is None:
            return self.__generate_candidate_samples(bounding_box, n)
        samples = []
        remain = n
        factor = 2
        while remain > 0 and factor < 16:
            samples.extend(
                _filter_candidate_samples(
                    self.__generate_candidate_samples(centered_box, remain * factor),
                    bounding_box,
                    overlap_range,
                    scale_range,
                    remain,
                )
            )
            remain = n - len(samples)
            factor = factor * 2
        return samples

    def _generate_sample_locations(self, bounding_box, n):
        raise NotImplementedError

    def __generate_candidate_samples(
        self, sample_center: numpy.array, n: int
    ) -> numpy.array:
        """
        Generate a set of candidate samples.

        :param numpy.array sample_center: The bounding box around which to sample. The (X,Y)
            coordinate for this box must specify the center of the box.
        :param int n: The number of candidate samples to generate.
        :return: A set of candidate samples.
        :rtype: numpy.array
        """
        candidate_samples = numpy.tile(sample_center[None, :], (n, 1))
        if self.aspect_ratio is not None:
            ratio = numpy.random.rand(n, 2) * 2 - 1
            candidate_samples[:, 2:] *= self.aspect_ratio ** ratio

        # TODO Invoke _gen_samples()

        # adjust box range
        candidate_samples[:, 2:] = numpy.clip(
            candidate_samples[:, 2:], 10, self.image_size - 10
        )
        if self.valid:
            candidate_samples[:, :2] = numpy.clip(
                candidate_samples[:, :2],
                candidate_samples[:, 2:] / 2,
                self.image_size - candidate_samples[:, 2:] / 2 - 1,
            )
        else:
            candidate_samples[:, :2] = numpy.clip(
                candidate_samples[:, :2], 0, self.image_size
            )

        # (min_x, min_y, w, h)
        candidate_samples[:, :2] -= candidate_samples[:, 2:] / 2
        return candidate_samples


class GaussianGenerator(Generator):
    """
    Generate image samples using a Gaussian distribution.
    """

    def _generate_sample_locations(
        self, centered_box: numpy.array, candidate_samples: numpy.array
    ) -> numpy.array:
        """
        Generate a set of samples using a Gaussian distribution.

        :param numpy.array centered_box: The bounding box around which to sample. The box's (X,Y)
            coordinates should specify the center of the box.
        :param candidate_samples: The candidate samples 
        :type candidate_samples: numpy.array
        :return: [description]
        :rtype: numpy.array
        """
        n = len(candidate_samples)
        candidate_samples[:, :2] += (
            self.translation
            * numpy.mean(utilities.bounding_boxes.dimensions(centered_box))
            * numpy.clip(0.5 * numpy.random.randn(n, 2), -1, 1)
        )
        candidate_samples[:, 2:] *= self.scale ** numpy.clip(
            0.5 * numpy.random.randn(n, 1), -1, 1
        )
        return candidate_samples


class UniformGenerator(Generator):
    """Generate image samples using a uniform distribution."""

    def _gen_samples(self, bounding_box, n: int):
        samples = numpy.tile(sample[None, :], (n, 1))

        if self.aspect_ratio is not None:
            ratio = numpy.random.rand(n, 2) * 2 - 1
            samples[:, 2:] *= self.aspect_ratio ** ratio

        # Uniform specific ------------------------------------------------------------------------
        # TODO Change these to functions.
        samples[:, :2] += (
            self.translation
            * numpy.mean(bounding_box[2:])
            * (numpy.random.rand(n, 2) * 2 - 1)
        )
        samples[:, 2:] *= self.scale ** (numpy.random.rand(n, 1) * 2 - 1)
        # Uniform specific ------------------------------------------------------------------------

        # adjust box range
        samples[:, 2:] = numpy.clip(samples[:, 2:], 10, self.image_size - 10)
        if self.valid:
            samples[:, :2] = numpy.clip(
                samples[:, :2],
                samples[:, 2:] / 2,
                self.image_size - samples[:, 2:] / 2 - 1,
            )
        else:
            samples[:, :2] = numpy.clip(samples[:, :2], 0, self.image_size)

        # (min_x, min_y, w, h)
        samples[:, :2] -= samples[:, 2:] / 2
        return samples


def generate_samples(
    generator: Generator,
    bounding_box: numpy.array,
    n: int,
    overlap_range: Iterable[float],
) -> numpy.array:
    """
    Generate a set of samples around the given bounding box.

    :param SampleGenerator generator_type: The generator to use for sample generation.
    :param numpy.array bounding_box: The bounding box around which to generate samples. This must be an
        array of four values: [x, y, width, height].
    :param int n: The number of samples to generate.
    :param Iterable[float] overlap_range: Two values describing the minimum and maximum allowed
        overlap with the ``bounding_box``.
    :returns: A 2D array of samples around the ``bounding_box``. The outer dimension is the
        individual samples. The inner dimension is the sample data: [x, y, width, height].
    :rtype: numpy.array

    This function is primarily syntactic sugar for generating samples using a one-shot generator.
    """
    return generator(bounding_box, n, overlap_range)


def _filter_candidate_samples(
    candidate_samples: numpy.array,
    bounding_box: numpy.array,
    overlap_range: Iterable[float],
    scale_range: Iterable[float],
    maximum_samples: int,
):
    """
    Filter out candidate samples based on the overlap and scale ranges.

    Arguments:
        candidate_samples {numpy.array} -- [description]
        bounding_box {numpy.array} -- [description]
        overlap_range {Iterable[float]} -- [description]
        scale_range {Iterable[float]} -- [description]
        maximum_samples {int} -- [description]

    Returns:
        [type] -- [description]
    """
    valid_sample_indices = numpy.ones(len(candidate_samples), dtype=bool)
    if overlap_range is not None:
        r = overlap_ratio(candidate_samples, bounding_box)
        valid_sample_indices *= (r >= overlap_range[0]) * (r <= overlap_range[1])
    if scale_range is not None:
        s = numpy.prod(candidate_samples[:, 2:], axis=1) / numpy.prod(bounding_box[2:])
        valid_sample_indices *= (s >= scale_range[0]) * (s <= scale_range[1])
    candidate_samples = candidate_samples[valid_sample_indices, :]
    return candidate_samples[: min(maximum_samples, len(candidate_samples))]


def _initialize_candidate_samples(
    sample_center: numpy.array, n: int, aspect_ratio
) -> numpy.array:
    """
    Create an array of initial candidate samples.

    :param numpy.array sample_center: The bounding box around which to sample.
    :param int n: The number of candidate samples to create.
    :param aspect_ratio: Aspect ratio adjust of the candidate samples. If
        this is ``None``, then no aspect ratio adjustment is performed.
    :return: Initial candidate samples.
    :rtype: numpy.array
    """
    candidate_samples = numpy.tile(sample_center[None, :], (n, 1))
    if aspect_ratio is not None:
        ratio = numpy.random.rand(n, 2) * 2 - 1
        candidate_samples[:, 2:] *= aspect_ratio ** ratio
    return candidate_samples
