"""
Convenience functionality for command line parsing.
"""

import argparse
import os.path


class PathSanitizer(argparse.Action):
    """Ensure path arguments are absolute and expand ~."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))


def add_dataset_path(parser: argparse.ArgumentParser, default: str) -> None:
    """
    Add the --dataset-path option to the parser.

    Args:
        parser (argparse.ArgumentParser): The parser to hold the --dataset-path option.
        default (str): The default value for the option. This function converts it to an
            absolute path before adding the option to the ``parser``.
    """
    parser.add_argument(
        "--dataset-path",
        help="The path to the dataset on disk.",
        default=os.path.abspath(os.path.expanduser(default)),
        action=PathSanitizer,
    )


def add_result_path(parser: argparse.ArgumentParser) -> None:
    """
    Add the --result-path option to a command line parser.

    Args:
        parser (argparse.ArgumentParser): The parser to hold the --result-path option.
    """
    parser.add_argument(
        "--result-path",
        help="The path to write tracking results.",
        default=os.path.abspath("./results"),
        action=PathSanitizer,
    )


def add_report_path(parser: argparse.ArgumentParser) -> None:
    """
    Add the --report-path option to a command line parser.

    Args:
        parser (argparse.ArgumentParser): The parser to hold the --report-path option.
    """
    parser.add_argument(
        "--report-path",
        help="The path to write experiment reports.",
        default=os.path.abspath("./reports"),
        action=PathSanitizer,
    )


def add_name_option(parser: argparse.ArgumentParser) -> None:
    """
    Add the --tracker-name option to a command line parser.

    Args:
        parser (argparse.ArgumentParser): The parser to hold the --tracker-name option.
    """
    parser.add_argument(
        "--tracker-name",
        help="The tracker name to use in experiment results and reports.",
    )


def add_slack_option(parser: argparse.ArgumentParser) -> None:
    """
    Add the --slack-file option to a command line parser.

    Args:
        parser (argparse.ArgumentParser): The command line parser to hold the option.
    """
    parser.add_argument(
        "--slack-file",
        help="The path to a Slack channel configuration file. With this option, the experiment "
        "script will send notifications to the channel in the file.",
        action=PathSanitizer,
    )
