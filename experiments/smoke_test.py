"""Run a smoke test using a single sequence from OTB-100."""

import time
import numpy
import PIL.Image
import torch
import yaml
import modules.utils
import tracking.gen_config
import tracking.tmft


def run(sequence_name: str) -> None:
    """
    Run a smoke test on a single sequence.

    Args:
        sequence_name (str): The name of the OTB-100 sequence to run. This is case sensitive.
    """
    print("Running the smoke test on the", sequence_name, "sequence.")
    with open("tracking/options.yaml") as yaml_file:
        configuration = yaml.safe_load(yaml_file)
    # For the smoke test, ensure the random generators are seeded. This makes the smoke test
    # deterministic; if the test fails, we KNOW it's from our code changes instead of randomness.
    numpy.random.seed(0)
    torch.manual_seed(0)
    _run_tmft(tracking.tmft.Tmft(configuration), sequence_name)


def _run_tmft(tmft: tracking.tmft, sequence: str) -> None:
    class _Arguments:
        def __init__(self, sequence_name) -> None:
            self.seq = sequence_name
            self.json = ""
            self.display = False
            self.savefig = False

    images, _, groundtruth, _, _, _, _, _ = tracking.gen_config.gen_config(
        _Arguments(sequence)
    )
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
