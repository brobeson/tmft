"""
modules.sample_generator
========================

Generate samples from an image.
"""

from typing import Iterable
import numpy as np
from PIL import Image

from .utils import overlap_ratio


class SampleGenerator:
    # TODO This should be a class hierarchy, not use a ``type_`` parameters."""
    def __init__(self, type_, img_size, trans=1, scale=1, aspect=None, valid=False):
        self.type = type_
        self.img_size = np.array(img_size)  # (w, h)
        self.trans = trans
        self.scale = scale
        self.aspect = aspect
        self.valid = valid

    def _gen_samples(self, bb, n):
        #
        # bb: target bbox (min_x,min_y,w,h)
        bb = np.array(bb, dtype="float32")

        # (center_x, center_y, w, h)
        sample = np.array(
            [bb[0] + bb[2] / 2, bb[1] + bb[3] / 2, bb[2], bb[3]], dtype="float32"
        )
        samples = np.tile(sample[None, :], (n, 1))

        # vary aspect ratio
        if self.aspect is not None:
            ratio = np.random.rand(n, 2) * 2 - 1
            samples[:, 2:] *= self.aspect ** ratio

        # sample generation
        if self.type == "gaussian":
            samples[:, :2] += (
                self.trans
                * np.mean(bb[2:])
                * np.clip(0.5 * np.random.randn(n, 2), -1, 1)
            )
            samples[:, 2:] *= self.scale ** np.clip(0.5 * np.random.randn(n, 1), -1, 1)

        elif self.type == "uniform":
            samples[:, :2] += (
                self.trans * np.mean(bb[2:]) * (np.random.rand(n, 2) * 2 - 1)
            )
            samples[:, 2:] *= self.scale ** (np.random.rand(n, 1) * 2 - 1)

        elif self.type == "whole":
            m = int(2 * np.sqrt(n))
            xy = np.dstack(
                np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))
            ).reshape(-1, 2)
            xy = np.random.permutation(xy)[:n]
            samples[:, :2] = bb[2:] / 2 + xy * (self.img_size - bb[2:] / 2 - 1)
            samples[:, 2:] *= self.scale ** (np.random.rand(n, 1) * 2 - 1)

        # adjust bbox range
        samples[:, 2:] = np.clip(samples[:, 2:], 10, self.img_size - 10)
        if self.valid:
            samples[:, :2] = np.clip(
                samples[:, :2],
                samples[:, 2:] / 2,
                self.img_size - samples[:, 2:] / 2 - 1,
            )
        else:
            samples[:, :2] = np.clip(samples[:, :2], 0, self.img_size)

        # (min_x, min_y, w, h)
        samples[:, :2] -= samples[:, 2:] / 2

        return samples

    def __call__(self, bbox, n, overlap_range=None, scale_range=None):
        if overlap_range is None and scale_range is None:
            samples = self._gen_samples(bbox, n)
        else:
            samples = None
            remain = n
            factor = 2
            while remain > 0 and factor < 16:
                samples_ = self._gen_samples(bbox, remain * factor)

                idx = np.ones(len(samples_), dtype=bool)
                if overlap_range is not None:
                    r = overlap_ratio(samples_, bbox)
                    idx *= (r >= overlap_range[0]) * (r <= overlap_range[1])
                if scale_range is not None:
                    s = np.prod(samples_[:, 2:], axis=1) / np.prod(bbox[2:])
                    idx *= (s >= scale_range[0]) * (s <= scale_range[1])

                samples_ = samples_[idx, :]
                samples_ = samples_[: min(remain, len(samples_))]
                if samples is None:
                    samples = samples_
                else:
                    samples = np.concatenate([samples, samples_])
                remain = n - len(samples)
                factor = factor * 2
        if len(samples) == 0:
            samples = np.full((n, len(bbox)), bbox)
        return samples

    def set_type(self, type_):
        self.type = type_

    def set_trans(self, trans):
        self.trans = trans

    def expand_trans(self, trans_limit):
        self.trans = min(self.trans * 1.1, trans_limit)


# ==================================================================================================
#                                                                          New code, already clean.
# ==================================================================================================
class Generator:
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
        if overlap_range is None and scale_range is None:
            return self._generate_samples(bounding_box, n)
        samples = []
        remain = n
        factor = 2
        while remain > 0 and factor < 16:
            samples.extend(
                _filter_candidate_samples(
                    self._generate_samples(bounding_box, remain * factor),
                    bounding_box,
                    overlap_range,
                    scale_range,
                    remain,
                )
            )
            # candidate_samples = self._generate_samples(bounding_box, remain * factor)
            # valid_sample_indices = np.ones(len(candidate_samples), dtype=bool)
            # if overlap_range is not None:
            #     r = overlap_ratio(candidate_samples, bounding_box)
            #     valid_sample_indices *= (r >= overlap_range[0]) * (r <= overlap_range[1])
            # if scale_range is not None:
            #     s = np.prod(candidate_samples[:, 2:], axis=1) / np.prod(bounding_box[2:])
            #     valid_sample_indices *= (s >= scale_range[0]) * (s <= scale_range[1])
            # candidate_samples = candidate_samples[valid_sample_indices, :]
            # candidate_samples = candidate_samples[: min(remain, len(candidate_samples))]
            # if samples is None:
            #     samples = candidate_samples
            # else:
            #     samples = np.concatenate([samples, candidate_samples])
            remain = n - len(samples)
            factor = factor * 2
        return samples

    def _generate_samples(self, bounding_box, n):
        raise NotImplementedError


def _filter_candidate_samples(
    candidate_samples: np.array,
    bounding_box: np.array,
    overlap_range: Iterable[float],
    scale_range: Iterable[float],
    maximum_samples: int,
):
    """
    Filter out candidate samples based on the overlap and scale ranges.

    Arguments:
        candidate_samples {np.array} -- [description]
        bounding_box {np.array} -- [description]
        overlap_range {Iterable[float]} -- [description]
        scale_range {Iterable[float]} -- [description]
        maximum_samples {int} -- [description]

    Returns:
        [type] -- [description]
    """
    valid_sample_indices = np.ones(len(candidate_samples), dtype=bool)
    if overlap_range is not None:
        r = overlap_ratio(candidate_samples, bounding_box)
        valid_sample_indices *= (r >= overlap_range[0]) * (r <= overlap_range[1])
    if scale_range is not None:
        s = np.prod(candidate_samples[:, 2:], axis=1) / np.prod(bounding_box[2:])
        valid_sample_indices *= (s >= scale_range[0]) * (s <= scale_range[1])
    candidate_samples = candidate_samples[valid_sample_indices, :]
    return candidate_samples[: min(maximum_samples, len(candidate_samples))]


class GuassianGenerator(Generator):
    def generate_samples(self, bounding_box, n: int):
        samples[:, :2] += (
            self.trans * np.mean(bb[2:]) * np.clip(0.5 * np.random.randn(n, 2), -1, 1)
        )
        samples[:, 2:] *= self.scale ** np.clip(0.5 * np.random.randn(n, 1), -1, 1)


def generate_samples(
    generator: SampleGenerator,
    bounding_box: np.array,
    n: int,
    overlap_range: Iterable[float],
) -> np.array:
    """
    Generate a set of samples around the given bounding box.

    :param SampleGenerator generator_type: The generator to use for sample generation.
    :param np.array bounding_box: The bounding box around which to generate samples. This must be an
        array of four values: [x, y, width, height].
    :param int n: The number of samples to generate.
    :param Iterable[float] overlap_range: Two values describing the minimum and maximum allowed
        overlap with the ``bounding_box``.
    :returns: A 2D array of samples around the ``bounding_box``. The outer dimension is the
        individual samples. The inner dimension is the sample data: [x, y, width, height].
    :rtype: np.array

    This function is primarily syntactic sugar for generating samples using a one-shot generator.
    """
    return generator(bounding_box, n, overlap_range)
