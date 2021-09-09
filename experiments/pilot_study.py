"""Run a pilot study using four sequences from OTB-100."""

import os.path
import time
import numpy
import PIL.Image
import torch
import yaml
import got10k.datasets
import modules.utils
import tracking.gen_config
import tracking.tmft


def main() -> None:
    """The main entry point of the pilot study application."""
    dataset = got10k.datasets.OTB(os.path.expanduser("~/Videos/otb"), version="tb100")
    for sequence in ["Car4", "Car24", "FleetFace", "Jump"]:
        run(sequence, dataset)


def run(sequence_name: str, dataset: got10k.datasets.OTB) -> None:
    """
    Run a smoke test on a single sequence.

    Args:
        sequence_name (str): The name of the OTB-100 sequence to run. This is case sensitive.
        dataset (got10k.datasets.OTB): The OTB-100 dataset.
    """
    print("Running the pilot study.")
    with open("tracking/options.yaml") as yaml_file:
        configuration = yaml.safe_load(yaml_file)
    # For the pilot study, ensure the random generators are seeded. This makes the study
    # deterministic; if the test fails, we KNOW it's from our code changes instead of randomness.
    numpy.random.seed(0)
    torch.manual_seed(0)
    _run_tmft(tracking.tmft.Tmft(configuration), sequence_name, dataset)


def _run_tmft(tmft: tracking.tmft, sequence: str, dataset: got10k.datasets.OTB) -> None:
    images, groundtruth = dataset[sequence]
    print("Training on frame 0")
    tmft.initialize(_load_image(images[0]), groundtruth[0])
    ious = numpy.zeros(len(images))
    frame_processing_times = numpy.zeros(len(images) - 1)
    ious[0] = 1.0
    for i, (image_file, gt) in enumerate(zip(images[1:], groundtruth[1:])):
        print("Frame", str(i + 1).rjust(3, " "), end="")
        start_time = time.time()
        target = tmft.find_target(_load_image(image_file))
        frame_processing_times[i] = time.time() - start_time
        ious[i + 1] = modules.utils.overlap_ratio(target, gt)
        print(f" IoU = {ious[i + 1]:.3f}  t = {frame_processing_times[i]:.3f}")
    print(f"Mean IoU = {ious.mean():.3f}")
    print(f"Mean t   = {frame_processing_times.mean():.3f}")


def _load_image(image_path: str):
    return PIL.Image.open(image_path).convert("RGB")


if __name__ == "__main__":
    main()
