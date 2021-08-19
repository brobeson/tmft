"""Provides the Tmft tracker class, and supporting classes and functions."""

import copy
from typing import List, Tuple
import numpy as np
import PIL.Image
import torch
import modules.model

from modules.model import BCELoss
from modules.sample_generator import SampleGenerator
from tracking.bbreg import BBRegressor
from tracking.run_tracker import forward_samples
from tracking.run_tracker import train


# TODO Do I need this function? I think it's to account for small targets in UAV123.
def _fix_positive_samples(samples: np.ndarray, n: int, box: np.ndarray) -> np.ndarray:
    """
    Ensure a set of positive samples is valid.

    :param numpy.ndarray samples: The positive samples to fix if invalid.
    :param int n: The number of samples to create if fixing is needed.
    :param numpy.ndarray box: The target bounding box used if fixing is needed.
    :return: The original samples if no fixing is required, or the fixed samples if fixing is
        required.
    :rtype: np.ndarray
    """
    if samples.shape[0] > 0:
        return samples
    return np.tile(box, [n, 1])


class BoundingBox:
    """Represents an axis aligned bounding box."""

    def __init__(self, x: int = 0, y: int = 0, width: int = 0, height: int = 0):
        self.__x = x
        self.__y = y
        self.__width = width
        self.__height = height

    @property
    def x(self):
        """The X coordinate of the upper left corner of the box, specified in pixels."""
        return self.__x

    @property
    def y(self):
        """The Y coordinate of the upper left corner of the box, specified in pixels."""
        return self.__y

    @property
    def width(self):
        """The width of the box, measured in pixels."""
        return self.__width

    @property
    def height(self):
        """The height of the box, measured in pixels."""
        return self.__height

    # TODO Add __str__()


def read_bounding_boxes(stream) -> List[BoundingBox]:
    """
    Read a list of bounding boxes from a file.

    :param stream: Any input stream that implements ``readlines()``.
    :return: The list of bounding boxes read from the ``stream``.
    :rtype: List[BoundingBox]
    """
    data = [l.strip().split(",") for l in stream.readlines()]
    return [
        BoundingBox(
            int(float(l[0])), int(float(l[1])), int(float(l[2])), int(float(l[3]))
        )
        for l in data
    ]


class Tmft:
    """
    The TMFT tracker.

    :param dict configuration: The TMFT configuration to use for this object. This is typically
        read from a YAML file.
    :raises TypeError: if the ``configuration`` is not a ``dict``.
    :raises ConfigurationError: if the ``configuration`` is not valid.
    """

    def __init__(self, configuration: dict):
        _validate_tmft_configuration(configuration)
        self.opts = copy.deepcopy(configuration)
        # TODO Which of these are really necessary as class members?
        self.cnn = None
        self.domain_network = None
        self.grl = None
        self.classification_loss = None
        self.domain_loss = None
        self.last_found_box = None
        self.box_regressor = None
        self.candidate_generator = None
        self.positive_training_generator = None
        self.negative_training_generator = None
        self.optimizer = None
        self.frame_number = 0
        # TODO Refactor these into a feature history object.
        self.positive_features = None
        self.negative_features = None
        self.training_records = []

    def initialize(self, image: PIL.Image.Image, ground_truth: BoundingBox) -> None:
        """
        Initialize the TMFT tracker.

        :param PIL.Image.Image image: The image to use to initialize the tracker. Typically, this is
            the first frame of a sequence, but some benchmarks may re-initialize the tracker if it
            loses the target. VOT is a benchmark that does this.
        :param BoundingBox ground_truth: The ground truth, axis-aligned bounding box for the target
            in the provided ``image``.

        This method will load the CNN from disk, then run the initial training with the provided
        ``image`` and ``ground_truth`` bounding box.
        """
        if "random_seed" in self.opts:
            np.random.seed(self.opts["random_seed"])
            torch.manual_seed(self.opts["random_seed"])

        self.cnn, self.domain_network = modules.model.make_networks(
            self.opts["model_path"],
            self.opts["use_gpu"],
            self.opts["ft_layers"],
            self.opts["grl"],
            **self.opts[self.opts["grl"]],
        )

        self.classification_loss = BCELoss()
        self.domain_loss = torch.nn.BCELoss(reduction="sum")
        self.optimizer = modules.model.make_optimizer(
            self.cnn, self.domain_network, self.opts["lr_init"], self.opts["lr_mult"]
        )
        self.last_found_box = ground_truth

        pos_feats, neg_feats = _generate_initial_features(
            self.opts, self.last_found_box, image, self.cnn
        )
        train(
            self.cnn,
            None,
            self.grl,
            self.classification_loss,
            None,
            self.optimizer,
            pos_feats,
            neg_feats,
            self.opts["maxiter_init"],
        )
        self.training_records.append([])
        torch.cuda.empty_cache()

        self.box_regressor = _initialize_bounding_box_regressor(
            self.opts, self.last_found_box, image, self.cnn
        )
        torch.cuda.empty_cache()

        (
            self.candidate_generator,
            self.positive_training_generator,
            self.negative_training_generator,
        ) = _initialize_tracking_sample_generators(self.opts, image.size)

        # Init pos/neg features for update
        neg_examples = self.negative_training_generator(
            self.last_found_box,
            self.opts["n_neg_update"],
            self.opts["overlap_neg_init"],
        )
        neg_feats = forward_samples(self.cnn, image, neg_examples)
        self.positive_features = [pos_feats]
        self.negative_features = [neg_feats]
        self.frame_number = 0
        self.optimizer = modules.model.make_optimizer(
            self.cnn, self.domain_network, self.opts["lr_update"], self.opts["lr_mult"]
        )

    def find_target(self, image: PIL.Image.Image) -> BoundingBox:
        """
        Attempt to locate the target object in an image.

        :param PIL.Image.Image image: The image in which to locate the target object.
        :return: The bounding box around the target object within the provided ``image``. If the
            target is not found, ``None`` is returned.
        :rtype: BoundingBox or NoneType

        This method will try to find the target object in the given ``image``. It also updates the
        CNN at the appropriate times.
        """

        self.frame_number = self.frame_number + 1
        if image.mode != "RGB":
            image = image.convert("RGB")
        target_found, target_bbox, best_samples = self.__find_target(image)
        self.__update_search_area(target_found)

        if target_found:
            regressed_target_box = self.__regress_target_box(best_samples, image)
        else:
            regressed_target_box = target_bbox

        if target_found:
            self.__collect_training_data(target_bbox, image)

        if not target_found:
            self.__short_term_update()
        elif self.frame_number % self.opts["long_interval"] == 0:
            self.__long_term_update()

        torch.cuda.empty_cache()
        self.last_found_box = target_bbox
        return regressed_target_box

    def __find_target(self, image: PIL.Image.Image) -> Tuple[bool, np.array, np.array]:
        """
        Find the target in the image.

        :param PIL.Image.Image image: The image in which to find the target.
        :returns: ``True`` if the target is found, ``False`` if it is not.
        :returns: The bounding box around the target.
        :returns: The samples the consitute the best five target candidates.
        """
        samples = self.candidate_generator(self.last_found_box, self.opts["n_samples"])
        sample_scores = forward_samples(self.cnn, image, samples, out_layer="fc6")
        top_scores, top_indices = sample_scores[:, 1].topk(5)
        top_indices = top_indices.cpu()
        target_score = top_scores.mean()
        target_bbox = samples[top_indices]
        if top_indices.shape[0] > 1:
            target_bbox = target_bbox.mean(axis=0)
        return target_score > 0, target_bbox, samples[top_indices]

    def __regress_target_box(
        self, best_samples: np.array, image: PIL.Image.Image
    ) -> np.array:
        """
        Submit the target bounding box to regression.

        :param np.array best_samples: The list of candidates that are most likely the target.
        :param PIL.Image.Image image: The image in which to regress the target bounding box.
        :return: The regressed bounding box.
        :rtype: np.array
        """
        if len(best_samples) == 1:
            best_samples = best_samples[None, :]
        best_samples = self.box_regressor.predict(
            forward_samples(self.cnn, image, best_samples), best_samples
        )
        return best_samples.mean(axis=0)

    def __collect_training_data(
        self, target_box: np.array, image: PIL.Image.Image
    ) -> None:
        """
        Collect positive and negative data for online training later.

        :param np.array target_box: The box around which to collect training data.
        :param PIL.Image.Image image: The image from which to collect training data.
        """
        self.positive_features.append(
            forward_samples(
                self.cnn,
                image,
                self.positive_training_generator(
                    target_box,
                    self.opts["n_pos_update"],
                    self.opts["overlap_pos_update"],
                ),
            )
        )
        if len(self.positive_features) > self.opts["n_frames_long"]:
            del self.positive_features[0]
        self.negative_features.append(
            forward_samples(
                self.cnn,
                image,
                self.negative_training_generator(
                    target_box,
                    self.opts["n_neg_update"],
                    self.opts["overlap_neg_update"],
                ),
            )
        )
        if len(self.negative_features) > self.opts["n_frames_short"]:
            del self.negative_features[0]

    def __short_term_update(self):
        number_of_frames = min(self.opts["n_frames_short"], len(self.positive_features))
        train(
            self.cnn,
            None,
            self.grl,
            self.classification_loss,
            self.domain_loss,
            self.optimizer,
            torch.cat(self.positive_features[-number_of_frames:], 0),
            torch.cat(self.negative_features, 0),
            self.opts["maxiter_update2"],
        )

    def __long_term_update(self):
        train(
            self.cnn,
            self.domain_network,
            self.grl,
            self.classification_loss,
            self.domain_loss,
            self.optimizer,
            torch.cat(self.positive_features, 0),
            torch.cat(self.negative_features, 0),
            self.opts["maxiter_update"],
        )

    def __update_search_area(self, target_found: bool):
        """
        Update the target search area based on whether the target was found this time.

        :param bool target_found: ``True`` indicates that the target was found in the current frame.
            ``False`` indicates that it was not found.
        """
        if target_found:
            self.candidate_generator.set_trans(self.opts["trans"])
        else:
            self.candidate_generator.expand_trans(self.opts["trans_limit"])


class ConfigurationError(Exception):
    """Indicates an error with the TMFT configuration."""

    def __init__(self, missing_key: str):
        super().__init__(
            f"{missing_key} is required, but not in the TMFT configuration."
        )


def open_image(path: str) -> PIL.Image.Image:
    """
    Load an image from disk.

    :param str path: The path to the image file on disk.
    :return: The requested image in RGB format.
    :rtype: PIL.Image.image
    """
    image = PIL.Image.open(path)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    return image


# -------------------------------------
# Implementation Details
# -------------------------------------
_REQUIRED_OPTIONS = [
    "use_gpu",
    "model_path",
    "img_size",
    "padding",
    "batch_pos",
    "batch_neg",
    "batch_neg_cand",
    "batch_test",
    "n_samples",
    "trans",
    "scale",
    "trans_limit",
    "trans_pos",
    "scale_pos",
    "trans_neg_init",
    "scale_neg_init",
    "trans_neg",
    "scale_neg",
    "n_bbreg",
    "overlap_bbreg",
    "trans_bbreg",
    "scale_bbreg",
    "aspect_bbreg",
    "lr_init",
    "maxiter_init",
    "n_pos_init",
    "n_neg_init",
    "overlap_pos_init",
    "overlap_neg_init",
    "lr_update",
    "maxiter_update",
    "maxiter_update2",
    "n_pos_update",
    "n_neg_update",
    "overlap_pos_update",
    "overlap_neg_update",
    "long_interval",
    "n_frames_long",
    "n_frames_short",
    "grad_clip",
    "grad_clip2",
    "lr_mult",
    "ft_layers",
    "loss_factor",
    "loss_factor2",
]


def _validate_tmft_configuration(configuration: dict) -> None:
    """
    Ensure that a TMFT configuration is valid.

    :param dict configuration: The configuration to validate.
    :raises TypeError: if ``configuration`` is not a ``dict``.
    """
    if not isinstance(configuration, dict):
        raise TypeError("The TMFT configuration must be a dict object.")
    for option in _REQUIRED_OPTIONS:
        if option not in configuration:
            raise ConfigurationError(option)


def _generate_initial_features(
    configuration: dict,
    bounding_box: np.array,
    image: PIL.Image.Image,
    cnn: modules.model.MDNet,
) -> Tuple[np.array, np.array]:
    """
    Create the initial set of features.

    :param dict configuration: The TMFT configuration.
    :param np.array bounding_box: The bounding box around which to sample and generate features.
    :param PIL.Image.Image image: The image from which to sample and generate features.
    :param modules.model.MDNet: The MDNet CNN from which to generate features.
    :return: The positive and negative features.
    :rtype: Tuple[np.array, np.array]
    """
    positive_samples = modules.sample_generator.generate_samples(
        SampleGenerator(
            "gaussian",
            image.size,
            configuration["trans_pos"],
            configuration["scale_pos"],
        ),
        bounding_box,
        configuration["n_pos_init"],
        configuration["overlap_pos_init"],
    )
    neg_examples = np.concatenate(
        [
            modules.sample_generator.generate_samples(
                SampleGenerator(
                    "uniform",
                    image.size,
                    configuration["trans_neg_init"],
                    configuration["scale_neg_init"],
                ),
                bounding_box,
                int(configuration["n_neg_init"] * 0.5),
                configuration["overlap_neg_init"],
            ),
            modules.sample_generator.generate_samples(
                SampleGenerator("whole", image.size),
                bounding_box,
                int(configuration["n_neg_init"] * 0.5),
                configuration["overlap_neg_init"],
            ),
        ]
    )
    neg_examples = np.random.permutation(neg_examples)
    return (
        forward_samples(cnn, image, positive_samples),
        forward_samples(cnn, image, neg_examples),
    )


def _initialize_bounding_box_regressor(
    configuration: dict,
    bounding_box: np.array,
    image: PIL.Image.Image,
    cnn: modules.model.MDNet,
) -> BBRegressor:
    """
    Create and train the bounding box regressor.

    :param dict configuration: The TMFT configuration.
    :param np.array bounding_box: The bounding box from which to train the regressor.
    :param PIL.Image.Image image: The image from which to sample and generate features.
    :param modules.model.MDNet: The MDNet CNN from which to generate features.
    :return: The trained bounding box regressor.
    :rtype: BBRegressor
    """
    samples = modules.sample_generator.generate_samples(
        SampleGenerator(
            "uniform",
            image.size,
            configuration["trans_bbreg"],
            configuration["scale_bbreg"],
            configuration["aspect_bbreg"],
        ),
        bounding_box,
        configuration["n_bbreg"],
        configuration["overlap_bbreg"],
    )
    regressor = BBRegressor(image.size)
    regressor.train(forward_samples(cnn, image, samples), samples, bounding_box)
    return regressor


def _initialize_tracking_sample_generators(configuration: dict, image_size: int):
    """
    Create the sample generators needed during tracking.

    :param dict configuration: The TMFT configuration.
    :param int image_size: The size of the image from which to draw samples.
    :return: The tracking sample generator, the training positive sample generator, and the training
        negative sample generator.
    :rtype: Tuple[SampleGenerator, SampleGenerator, SampleGenerator]
    """
    tracking_generator = SampleGenerator(
        "gaussian", image_size, configuration["trans"], configuration["scale"]
    )
    positive_generator = SampleGenerator(
        "gaussian", image_size, configuration["trans_pos"], configuration["scale_pos"]
    )
    negative_generator = SampleGenerator(
        "uniform", image_size, configuration["trans_neg"], configuration["scale_neg"]
    )
    return tracking_generator, positive_generator, negative_generator
