"""Run VOT experiments."""

import sys
import numpy
import PIL.Image
import torch
import yaml
import modules.utils
import tracking.trackerADMDNet
import tracking.gen_config
import tracking.tmft


class Arguments:
    """This is necessary because gen_config.gen_config() is poorly engineered."""

    def __init__(self, sequence_name) -> None:
        self.seq = sequence_name
        self.json = ""
        self.display = False
        self.savefig = False


def run_tmft(tmft: tracking.tmft, sequence: str) -> None:
    """
    Run the TMFT tracker on a sequence.

    Args:
        tmft (tracking.tmft): The TMFT tracker to run.
        sequence (str): The name of the sequence to track.
    """
    images, _, groundtruth, _, _, _, _, _ = tracking.gen_config.gen_config(
        Arguments(sequence)
    )
    print("Tracking", sequence)
    print("Training on frame 0")
    tmft.initialize(_load_image(images[0]), groundtruth[0])
    ious = numpy.zeros(len(images))
    ious[0] = 1.0
    for i, (image_file, gt) in enumerate(zip(images[1:], groundtruth[1:])):
        print("Frame", i + 1, end="")
        target = tmft.find_target(_load_image(image_file))
        ious[i + 1] = modules.utils.overlap_ratio(target, gt)
        print(f" IoU = {ious[i + 1]:.3f}")
    print(f"Mean IoU = {ious.mean():.3f}")


def main() -> None:
    """The main entry for the OTB experiments."""
    with open("tracking/options.yaml") as yaml_file:
        configuration = yaml.safe_load(yaml_file)
    if "random_seed" in configuration:
        numpy.random.seed(configuration["random_seed"])
        torch.manual_seed(configuration["random_seed"])
    if len(sys.argv) > 1 and sys.argv[1] == "admdnet":
        run_tmft(tracking.trackerADMDNet.ADMDNet("ADMDNet", configuration), "Deer")
    else:
        run_tmft(tracking.tmft.Tmft(configuration), "Deer")


def _load_image(image_path: str):
    return PIL.Image.open(image_path).convert("RGB")


if __name__ == "__main__":
    main()
