"""
Run OTB experiments.

Copyright brobeson
"""

import argparse
import os.path
import got10k.experiments
import experiments.got10k_wrapper
import tracking.tmft


def main() -> None:
    """The main entry point of the OTB experiment runner."""
    arguments = _parse_command_line()
    experiment = got10k.experiments.ExperimentOTB(
        arguments.dataset_path, version=arguments.version
    )
    tracker = experiments.got10k_wrapper.Got10kTmft(
        tracking.tmft.Tmft(
            tracking.tmft.read_configuration(
                os.path.expanduser("~/repositories/tmft/tracking/options.yaml")
            )
        )
    )
    experiment.run(tracker)


def _parse_command_line() -> argparse.Namespace:
    """
    Parse the command line.

    Returns:
        argparse.Namespace: The command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run OTB experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        help="The path to the dataset on disk.",
        default=os.path.expanduser("~/Videos/otb"),
    )
    parser.add_argument(
        "--version",
        help="The OTB dataset to use.",
        choices=["tb50", "tb100"],
        default="tb100",
    )
    arguments = parser.parse_args()
    arguments.dataset_path = os.path.abspath(os.path.expanduser(arguments.dataset_path))
    return arguments


# def _run_multiple_lr_schedules() -> None:
#     configurations = [
#         {"grl": "constant"},
#         {"grl_direction": "decreasing", "grl": "arccosine"},
#         {"grl_direction": "decreasing", "grl": "cosine_annealing"},
#         {"grl_direction": "decreasing", "grl": "exponential"},
#         {"grl_direction": "decreasing", "grl": "gamma"},
#         {"grl_direction": "decreasing", "grl": "linear"},
#         {"grl_direction": "decreasing", "grl": "pada"},
#         {"grl_direction": "increasing", "grl": "arccosine"},
#         {"grl_direction": "increasing", "grl": "cosine_annealing"},
#         {"grl_direction": "increasing", "grl": "exponential"},
#         {"grl_direction": "increasing", "grl": "gamma"},
#         {"grl_direction": "increasing", "grl": "linear"},
#         {"grl_direction": "increasing", "grl": "pada"},
#     ]
#     experiment = got10k.experiments.ExperimentUAV123(
#         os.path.expanduser("~/Videos/uav123")
#     )
#     tracker_names = []
#     for configuration in configurations:
#         if "grl_direction" in configuration:
#             name = configuration["grl_direction"][0:3] + "_" + configuration["grl"]
#         else:
#             name = configuration["grl"]
#         tracker_names.append(name)
#         tracker = tracking.tmft.ADMDNet(name, configuration)
#         experiment.run(tracker)
#     experiment.report(tracker_names)


if __name__ == "__main__":
    main()
