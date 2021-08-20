"""
TMFT Experiment Runner

This is the main entry module for running experiments on TMFT. This document is the reference for
the module. For basic instructions for how to run experiments, see :doc:`user_guide`.
"""

import argparse
import os.path

# import yaml
# import experiments.got10k_wrapper
import experiments.smoke_test


def main() -> None:
    """The main entry point for running any experiments."""
    arguments = parse_command_line()
    if arguments.smoke_test:
        experiments.smoke_test.run(arguments.smoke_test)
    # configuration = _load_configuration_from_file(arguments.configuration_file)
    # if "multirun" in configuration:
    #     _run_multiple_experiments(configuration["multirun"])
    #     return
    # _run_experiment(configuration)


def parse_command_line() -> argparse.Namespace:
    """
    Parse the command arguments.

    Returns:
        argparse.Namespace: The arguments and their values.
    """
    parser = argparse.ArgumentParser(
        description="Run all tracking experiments specified in a configuration file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # parser.add_argument(
    #     "--sequence",
    #     help="Run TMFT on a single sequence. Using this option skips all experiments defined in the"
    #     " configuration file. This option must specify the root path to a sequence from a known"
    #     " dataset. The experiment runner will attempt to figure out the dataset, but it will not"
    #     " attempt to run actual benchmarks.",
    # )
    parser.add_argument(
        "--smoke-test",
        help="Run a smoke test on a single OTB-100 sequence. The sequence name is case sensitive.",
        metavar="SEQUENCE",
    )
    parser.add_argument(
        "configuration_file",
        nargs="?",
        default="configuration.yaml",
        help="The path to the experiment configuration file.",
    )
    arguments = parser.parse_args()
    arguments.configuration_file = os.path.abspath(
        os.path.expanduser(arguments.configuration_file)
    )
    return arguments


# def _load_configuration_from_file(file_path: str) -> dict:
#     """
#     Read the experimental configuration from a file.

#     :param str file_path: The path to the configuration file to read.
#     :returns: The loaded configuration.
#     :rtype: dict
#     """
#     configuration = yaml.safe_load(open(file_path))
#     if "display" not in configuration:
#         configuration["display"] = False
#     return configuration


# def _should_run_smoke_test(configuration: dict) -> None:
#     return "smoke_test" in configuration and (
#         "skip" not in configuration["smoke_test"]
#         or not configuration["smoke_test"]["skip"]
#     )


# def _run_multiple_experiments(multi_configuration: dict) -> None:
#     """
#     Run experiments for multiple configurations.

#     :param dict multi_configuration: The configuration that specifies the actual experimental
#         configurations to run.
#     """
#     if "path" in multi_configuration:
#         directory = os.path.abspath(os.path.expanduser(multi_configuration["path"]))
#     else:
#         directory = os.path.abspath(".")
#     for configuration in multi_configuration["configurations"]:
#         if os.path.isabs(configuration):
#             configuration_file = configuration
#         else:
#             configuration_file = os.path.join(directory, configuration)
#         if not configuration_file.endswith(".yaml"):
#             configuration_file = configuration_file + ".yaml"
#         try:
#             _run_experiment(configuration_file)
#         except RuntimeError as error:
#             print(f"Skipping {configuration}:", error)


# def _run_experiment(configuration) -> None:
#     """
#     Run a single set of experiments defined in a configuration.

#     :param configuration: An actual configuration ``dict``, or the path to a configuration file.
#     """
#     if isinstance(configuration, str):
#         configuration = _load_configuration_from_file(configuration)
#     if "tmft" not in configuration:
#         raise RuntimeError("The 'tmft' section is missing from the configuration.")
#     print("Running", configuration["tmft"]["name"])
#     if _should_run_smoke_test(configuration):
#         experiments.single_sequence.run_smoke_test(configuration)
#         return
#     if "got10k_experiments" in configuration:
#         experiments.got10k_wrapper.run(configuration)


if __name__ == "__main__":
    main()
