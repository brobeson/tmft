"""
experiments.got10k
==================

Run experiments using the GOT-10k tool.
"""

import os.path
import sys
import got10k.datasets
import got10k.experiments
import got10k.trackers
import tracking.tmft
import utilities.loss_data


class Got10kTmft(got10k.trackers.Tracker):
    """A wrapper class so the GOT-10k tool can run TMFT."""

    def __init__(self, configuration: dict):
        super().__init__(
            name="TMFT" if "name" not in configuration else configuration["name"],
            is_deterministic=False,
        )
        self.tracker = tracking.tmft.Tmft(configuration)

    def init(self, image, box):
        self.tracker.initialize(image, box)

    def update(self, image):
        return self.tracker.find_target(image)


def run(configuration: dict):
    """The main entry point for the application."""
    if "tmft" not in configuration:
        raise KeyError("The key 'tmft' is required in the configuration.")
    if "display" not in configuration["got10k_experiments"]:
        configuration["got10k_experiments"]["display"] = False
    if "otb" in configuration["got10k_experiments"]:
        _run_otb_experiments(
            configuration["tmft"],
            configuration["got10k_experiments"]["otb"],
            configuration["got10k_experiments"]["display"],
        )
    if "vot" in configuration["got10k_experiments"]:
        # For VOT, we should not explicitely seed the random number generators.
        if "random_seed" in configuration["tmft"]:
            del configuration["tmft"]["random_seed"]
        _run_vot_experiments(
            configuration["tmft"],
            configuration["got10k_experiments"]["vot"],
            configuration["got10k_experiments"]["display"],
        )


def _write_otb_loss_data(
    result_directory: str, version: str, tracker_name: str, loss_data,
) -> None:
    """
    Write tracking loss data to disk.

    :param dict configuration: The full experiment configuration that was run.
    :param list loss_data: The loss data to write to disk.
    """
    directory = os.path.join(result_directory, "OTB" + version, tracker_name)
    os.makedirs(directory, exist_ok=True)
    for i, loss in enumerate(loss_data):
        utilities.loss_data.write_training_records(
            loss, os.path.join(directory, f"{i:03}.txt")
        )


def _sanitize_path(path: str) -> str:
    """
    Ensure that a path is absolute, and has home shortcuts expanded.

    :param str path: The path to be sanitized.
    :return: The provided ``path`` converted to an absolute path, and with '~' converted to an
        actual path string.
    :rtype: str
    """
    return os.path.abspath(os.path.expanduser(path))


def _run_vot_experiments(
    tmft_configuration: dict, configuration: dict, display: bool
) -> None:
    """
    Run VOT experiments.

    :param dict tmft_configuration: The configuration to use for the TMFT tracker.
    :param dict configuration: The experimental configuration for VOT run via GOT-10k.
    :param bool display: True indicates to display imagery as tracking occurs. False indicates to
        not display imagery. This can be overridden by ``configuration["display"]``.
    """
    if "skip" in configuration and configuration["skip"]:
        return
    if "root_dir" not in configuration:
        raise KeyError(
            "'got10k_experiments.vot.root_dir' is required in the experimental configuration."
        )
    if "version" not in configuration:
        raise KeyError(
            "'got10k_experiments.vot.version' is required in the experimental configuration."
        )
    experiment = got10k.experiments.ExperimentVOT(
        _sanitize_path(configuration["root_dir"]),
        version=configuration["version"],
        experiments="supervised",
        result_dir=_sanitize_path(configuration["result_dir"]),
    )
    tracker = Got10kTmft(tmft_configuration)
    experiment.run(
        tracker,
        visualize=("display" in configuration and configuration["display"])
        or ("display" not in configuration and display),
    )
    experiment.report([tracker.name])


def _run_otb_experiments(
    tmft_configuration: dict, configuration: dict, display: bool
) -> None:
    """
    Run OTB experiments.

    :param dict tmft_configuration: The configuration for the TMFT tracker.
    :param dict configuration: The experimental configuration for OTB run via GOT-10k.
    :param bool display: True indicates to display imagery as tracking occurs. False indicates to
        not display imagery. This can be overridden by ``configuration["display"]``.
    """
    if "skip" in configuration and configuration["skip"]:
        return
    if "root_dir" not in configuration:
        raise KeyError(
            "'got10k_experiments.otb.root_dir' is required in the experimental configuration."
        )
    if "version" not in configuration:
        raise KeyError(
            "'got10k_experiments.otb.version' is required in the experimental configuration."
        )
    if not isinstance(configuration["version"], list):
        configuration["version"] = [configuration["version"]]
    configuration["result_dir"] = _sanitize_path(configuration["result_dir"])
    for version in configuration["version"]:
        experiment = got10k.experiments.ExperimentOTB(
            _sanitize_path(configuration["root_dir"]),
            version=version,
            result_dir=configuration["result_dir"],
        )
        tracker = Got10kTmft(tmft_configuration)
        experiment.run(
            tracker,
            visualize=("display" in configuration and configuration["display"])
            or ("display" not in configuration and display),
        )
        experiment.report([tracker.name])
        _write_otb_loss_data(
            configuration["result_dir"],
            version,
            tmft_configuration["name"],
            tracker.tracker.training_records,
        )
