"""
This module provides convenient functionality for interacting with the terminal and the command
line.
"""

import argparse
import os.path


class PathSanitizer(argparse.Action):
    """
    Ensures path arguments are absolute and expands '~'.

    To use this, just pass it to `argparse.ArgumentParser.add_argument()
    <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument>`_
    as the ``action``.

    .. code-block:: python

        import experiments.command_line
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "file_path",
            action=experiments.command_line.PathSanitizer,
        )

    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))


def add_dataset_dir_parameter(parser: argparse.ArgumentParser, default: str) -> argparse.Action:
    """
    Add the ``--dataset-dir`` parameter to a command line parser.

    This allows the user to specify the location to find a benchmark's image and annotation files.

    Args:
        parser (argparse.ArgumentParser): Add the parameter to this parser.
        default (str): Use this as the default argument for the parameter. This function expands
            '~' and converts it to an absolute path before adding the option to the ``parser``.

    Returns:
        argparse.Action: This function returns the :py:class:`argparse.Action` that represents the
        command line argument. The caller can tweak the action if necessary.
    """
    return parser.add_argument(
        "--dataset-dir",
        help="This is the path to a tracking benchmark dataset. It is specific to one benchmark, "
        "such as OTB-100 or VOT 2019. The layout of the directory content is specific to each "
        "benchmark",
        default=os.path.abspath(os.path.expanduser(default)),
        action=PathSanitizer,
    )


def add_results_dir_parameter(parser: argparse.ArgumentParser) -> argparse.Action:
    """
    Add the ``--results-dir`` parameter to a command line parser.

    This allows the user to specify the location to read or write tracking experiment results files.

    Args:
        parser (argparse.ArgumentParser): Add the ``--results-dir`` parameter to this parser.

    Returns:
        argparse.Action: This function returns the :py:class:`argparse.Action` that represents the
        command line argument. The caller can tweak the action if necessary.
    """
    return parser.add_argument(
        "--results-dir",
        help="This is the path to the directory containing tracking results. Commands that "
        "generate tracking results write them to this directory in benchmark child directories. "
        "Commands that use tracking results read from this directory, from benchmark child "
        "directories.",
        default=os.path.abspath("./results"),
        action=PathSanitizer,
    )


def add_tracker_name_parameter(parser: argparse.ArgumentParser) -> argparse.Action:
    """
    Add the ``--tracker-name`` parameter to a command line parser.

    This allows the user to select a specific tracker. The purpose of the tracker name depends on
    the command.

    Args:
        parser (argparse.ArgumentParser): Add the parameter to this parser.

    Returns:
        argparse.Action: This function returns the :py:class:`argparse.Action` that represents the
        command line argument. The caller can tweak the action if necessary.
    """
    return parser.add_argument(
        "--tracker-name", help="Use this name for the tracker results.", default="TMFT"
    )


def print_information(*objects) -> None:
    """
    Print an information message to the terminal using blue text.

    Args:
        objects: The data to print as an information message.
    """
    _print_message("\033[94m", *objects)


def print_warning(*objects) -> None:
    """
    Print a warning message to the terminal using orange text.

    Args:
        objects: The data to print as a warning.
    """
    _print_message("\033[91m", *objects)


def _print_message(color: str, *objects) -> None:
    print(color, end="")
    print(*objects, end="")
    print("\033[0m")
