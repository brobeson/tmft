"""Run tracking experiments."""

from __future__ import absolute_import

import logging
import os.path
import sys
import yaml

import got10k.experiments

from tracking.trackerADMDNet import ADMDNet


def sanitize_path(path: str) -> str:
    """
    Ensure that a path is absolute, and has home shortcuts expanded.

    :param str path: The path to be sanitized.
    :return: The provided ``path`` converted to an absolute path, and with '~' converted to an
        actual path string.
    :rtype: str
    """
    return os.path.abspath(os.path.expanduser(path))


def run_vot_experiments(configuration: dict, display: bool) -> None:
    """
    Run VOT experiments.

    :param dict configuration: The experimental configuration for VOT run via GOT-10k.
    :param bool display: True indicates to display imagery as tracking occurs. False indicates to
        not display imagery. This can be overridden by ``configuration["display"]``.
    """
    if "skip" in configuration and configuration["skip"]:
        logging.info(
            "Skipping VOT experiments because 'skip: true' is in the configuration."
        )
        return
    if "root_dir" not in configuration:
        logging.warning(
            "'got10k_experiments.vot.root_dir' is required in the experimental configuration."
        )
        return
    if "version" not in configuration:
        logging.warning(
            "'got10k_experiments.vot.version' is required in the experimental configuration."
        )
        return
    experiment = got10k.experiments.ExperimentVOT(
        sanitize_path(configuration["root_dir"]),
        version=configuration["version"],
        experiments="supervised",
        result_dir=sanitize_path(configuration["result_dir"]),
    )
    tracker = ADMDNet()
    experiment.run(
        tracker,
        visualize=("display" in configuration and configuration["display"])
        or ("display" not in configuration and display),
    )
    experiment.report([tracker.name])


def run_otb_experiments(configuration: dict, display: bool) -> None:
    """
    Run OTB experiments.

    :param dict configuration: The experimental configuration for OTB run via GOT-10k.
    :param bool display: True indicates to display imagery as tracking occurs. False indicates to
        not display imagery. This can be overridden by ``configuration["display"]``.
    """
    if "skip" in configuration and configuration["skip"]:
        logging.info(
            "Skipping OTB experiments because 'skip: true' is in the configuration."
        )
        return
    if "root_dir" not in configuration:
        logging.warning(
            "'got10k_experiments.otb.root_dir' is required in the experimental configuration."
        )
        return
    if "version" not in configuration:
        logging.warning(
            "'got10k_experiments.otb.version' is required in the experimental configuration."
        )
        return
    experiment = got10k.experiments.ExperimentOTB(
        sanitize_path(configuration["root_dir"]),
        version=configuration["version"],
        result_dir=sanitize_path(configuration["result_dir"]),
    )
    tracker = ADMDNet()
    experiment.run(
        tracker,
        visualize=("display" in configuration and configuration["display"])
        or ("display" not in configuration and display),
    )
    experiment.report([tracker.name])


def main():
    """The main entry point for the application."""
    logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)
    configuration = yaml.safe_load(open("tracking/options.yaml", "r"))
    if "got10k_experiments" not in configuration:
        sys.exit("No GOT-10k experiments were in tracking/options.yaml")
    configuration = configuration["got10k_experiments"]
    if "display" not in configuration:
        configuration["display"] = False
    if "vot" in configuration:
        run_vot_experiments(
            configuration["vot"], configuration["display"],
        )
    if "otb" in configuration:
        run_otb_experiments(
            configuration["otb"], configuration["display"],
        )


if __name__ == "__main__":
    main()
