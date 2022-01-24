"""
Run a fast pilot study of a tracker.

This module runs a pilot tracking study. The primary use cases are rapid feedback of tracker
performance, and quick testing of changes to tracker code.

Running this Module as a Script
-------------------------------

You can run this module as a stand-alone script.

.. literalinclude:: generated/pilot_help.rst
    :language: text

Importing this Module
---------------------

You can also use this module as part of a larger application.

#. Import this module.
#. Call :py:func:`fill_command_line_parser()`.
#. Parse the command line arguments.
#. Run ``arguments.func(arguments)`` or :py:func:`main()`.

Here is an example of using this module as the sole command line parser::

    import experiments.pilot_study as pilot_study
    parser = pilot_study.fill_command_line_parser(argparse.ArgumentParser())
    arguments = parser.parse_args()
    arguments.func(arguments)

Here is an example of using this module as a subcommand in a larger application::

    import experiments.pilot_study as pilot_study
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    pilot_study.fill_command_line_parser(subparsers.add_parser("pilot_study"))
    arguments = parser.parse_args()
    arguments.func(arguments)

Reference
---------
"""

import argparse
import json
import os
import time
import numpy
import PIL.Image
import got10k.experiments
import experiments.command_line
import modules.utils
import tracking.gen_config
import tracking.tmft


def fill_command_line_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Create the command line parser for this module.

    This function supports filling in a subparser or a root parser. In both cases, this function
    overwrites certain parser attributes, such as the description.

    Args:
        parser (argparse.ArgumentParser): Fill out this argument parser. This can be a root parser
            or a subparser created with `add_subparsers()
            <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_subparsers>`_.

    Returns:
        The parser, filled with parameters and attributes, ready for command line parsing.
    """
    parser.description = (
        "Run tracking tests on a subset of OTB-100 sequences. This command is faster than running "
        "a full battery of tracking experiments. It saves the results in a JSON database; use the "
        "experiments.report module to analyze the results of the pilot study."
    )
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.set_defaults(func=main)
    experiments.command_line.add_tracker_name_parameter(parser)
    experiments.command_line.add_dataset_dir_parameter(parser, "~/Videos/otb")
    experiments.command_line.add_results_dir_parameter(parser)
    parser.add_argument(
        "sequences",
        help="Track this sequences in the pilot study. These must name a sequence in the OTB-100"
        "dataset and is case-sensitive. See "
        "http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html for available sequences.",
        nargs="+",
        metavar="sequence",
    )
    return parser


def main(arguments: argparse.Namespace) -> None:
    """
    The main entry point for this module.

    Typically, you don't need to invoke this function; instead use ``arguments.func()`` after you
    parse the command line arguments. See :py:func:`fill_command_line_parser()` for examples. If
    you do need to call this function, do so *after* parsing the command line.

    Args:
        arguments (argparse.Namespace): The parsed command line arguments. The ``arguments`` must
            have these attributes: ``sequences``, ``dataset_dir``, ``tracker_name``, and
            ``results_dir``.
    """
    progress_bar = _ProgressBar((len(max(arguments.sequences, key=len)) + 1, 0), 0, "")
    dataset = got10k.datasets.OTB(arguments.dataset_dir, version="tb100")
    results = {sequence: {} for sequence in arguments.sequences}
    for sequence in arguments.sequences:
        sequence_result = _run_sequence(sequence, dataset, progress_bar)
        results[sequence] = {
            "mean iou": sequence_result[0],
            "mean time": sequence_result[1],
        }
    for sequence, data in results.items():
        print(sequence)
        print(f"  Mean IoU = {data['mean iou']:.3f}")
        print(f"  Mean t   = {data['mean time']:.3f}")
    if arguments.tracker_name:
        _save_results(
            arguments.tracker_name,
            arguments.results_dir,
            {sequence: data["mean iou"] for sequence, data in results.items()},
        )


class _ProgressBar:
    """
    An object that can print a progress bar on the console.

    Attributes:
        margins (tuple): Left and right margins, respectively, measured in columns.
        maximum (int): The maximum value of the progress bar. The minimum is fixed at 0.
        label (str): A label to print before the bar. This must fit inside the left margin.
    """

    def __init__(self, margins: int, maximum: int, label: str):
        self.margins = margins
        self.maximum = float(maximum)
        self.label = label

    def print(self, i: int) -> None:
        """
        Print the progress bar on the console.

        Args:
            i (int): The current value of the progress bar.
        """
        bar_capacity = os.get_terminal_size()[0] - self.margins[0] - self.margins[1] - 2
        bar_width = int(float(i) / self.maximum * bar_capacity)
        space_width = bar_capacity - bar_width
        print(
            self.label,
            " " * (self.margins[0] - len(self.label)),
            "[",
            "=" * bar_width,
            " " * space_width,
            "]",
            sep="",
            end="\r",
        )


def _run_sequence(
    sequence_name: str, dataset: got10k.datasets.OTB, progress_bar: _ProgressBar
) -> None:
    # Ensure the random generators are seeded. This makes the study deterministic; if the test
    # fails, we KNOW it's from our code changes instead of randomness.
    tmft = tracking.tmft.Tmft(tracking.tmft.read_configuration("tracking/options.yaml"))
    tmft.opts["random_seed"] = 0
    images, groundtruth = dataset[sequence_name]
    progress_bar.label = sequence_name
    progress_bar.maximum = len(images)
    print("Initializing", sequence_name, "on frame 0...", end="\r")
    tmft.initialize(_load_image(images[0]), groundtruth[0])
    ious = numpy.zeros(len(images))
    frame_processing_times = numpy.zeros(len(images))
    ious[0] = 1.0
    for i, (image_file, gt) in enumerate(zip(images[1:], groundtruth[1:]), start=1):
        progress_bar.print(i)
        start_time = time.time()
        target = tmft.find_target(_load_image(image_file))
        frame_processing_times[i] = time.time() - start_time
        ious[i] = modules.utils.overlap_ratio(target, gt)
    progress_bar.print(progress_bar.maximum)
    print()
    return ious.mean(), frame_processing_times[1:].mean()


def _load_image(image_path: str) -> PIL.Image.Image:
    return PIL.Image.open(image_path).convert("RGB")


def _save_results(tracker_name: str, results_dir: str, results: dict) -> None:
    results_path = os.path.join(results_dir, "pilot_results.json")
    if os.path.isfile(results_path):
        with open(results_path, "r") as results_file:
            current_data = json.load(results_file)
    else:
        current_data = {}
    if tracker_name in current_data:
        current_data[tracker_name]["scores"] = results
    else:
        current_data[tracker_name] = {"scores": results, "tags": []}
    with open(results_path, "w") as results_file:
        json.dump(current_data, results_file, indent=2)


if __name__ == "__main__":
    PARSER = fill_command_line_parser(argparse.ArgumentParser())
    ARGUMENTS = PARSER.parse_args()
    ARGUMENTS.func(ARGUMENTS)
