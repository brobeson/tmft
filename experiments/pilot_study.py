"""Run a smoke test using a single sequence from OTB-100."""

import argparse
import json
import os
import time
import numpy
import PIL.Image
import torch
import got10k.experiments
import experiments.command_line
import modules.utils
import tracking.gen_config
import tracking.tmft


class ProgressBar:
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


def main() -> None:
    """The main entry point of the smoke test application."""
    arguments = _parse_command_line()
    progress_bar = ProgressBar((len(max(arguments.sequences, key=len)) + 1, 0), 0, "")
    dataset = got10k.datasets.OTB(arguments.dataset_path, version="tb100")
    results = {sequence: {} for sequence in arguments.sequences}
    for sequence in arguments.sequences:
        sequence_result = run(sequence, dataset, progress_bar)
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
            {sequence: data["mean iou"] for sequence, data in results.items()},
        )


def run(
    sequence_name: str, dataset: got10k.datasets.OTB, progress_bar: ProgressBar
) -> None:
    """
    Run a smoke test on a single sequence.

    Args:
        sequence_name (str): The name of the OTB-100 sequence to run. This is case sensitive.
        dataset (got10k.datasets.OTB): The OTB-100 dataset.
        progress_bar (ProgressBar): A progress bar to print on the console.
    """
    # Ensure the random generators are seeded. This makes the study deterministic; if the test
    # fails, we KNOW it's from our code changes instead of randomness.
    tmft = tracking.tmft.Tmft(tracking.tmft.read_configuration("tracking/options.yaml"))
    tmft.opts["random_seed"] = 0
    images, groundtruth = dataset[sequence_name]
    progress_bar.label = sequence_name
    progress_bar.maximum = len(images)
    print("Initializing", sequence_name, "on frame 0...", end="\r")
    tmft.initialize(load_image(images[0]), groundtruth[0])
    ious = numpy.zeros(len(images))
    frame_processing_times = numpy.zeros(len(images))
    ious[0] = 1.0
    for i, (image_file, gt) in enumerate(zip(images[1:], groundtruth[1:]), start=1):
        progress_bar.print(i)
        start_time = time.time()
        target = tmft.find_target(load_image(image_file))
        frame_processing_times[i] = time.time() - start_time
        ious[i] = modules.utils.overlap_ratio(target, gt)
    progress_bar.print(progress_bar.maximum)
    print()
    return ious.mean(), frame_processing_times[1:].mean()


def load_image(image_path: str) -> PIL.Image.Image:
    """
    Load a sequence image from disk.

    Args:
        image_path (str): The path to the image on disk.

    Returns:
        PIL.Image.Image: The sequence image in RGB format.
    """
    return PIL.Image.open(image_path).convert("RGB")


def _parse_command_line() -> argparse.Namespace:
    """Parse the comment line and return arguments."""
    parser = argparse.ArgumentParser(
        description="Run a smoke test using a single OTB-100 sequence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    experiments.command_line.add_name_option(parser)
    parser.add_argument(
        "sequences",
        help="The sequences to use for the smoke test. The sequence names are case sensitive.",
        nargs="+",
    )
    experiments.command_line.add_dataset_path(parser, "~/Videos/otb")
    arguments = parser.parse_args()
    return arguments


def _save_results(tracker_name: str, results: dict) -> None:
    results_path = "results/pilot_results.json"
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
    main()
