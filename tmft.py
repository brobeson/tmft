#!/usr/bin/env python3

"""The control application for TMFT experiments, reports, and more."""

import argparse
import experiments.experiment
import experiments.pilot_study
import experiments.report

# Parse the command line
PARSER = argparse.ArgumentParser()
SUBPARSERS = PARSER.add_subparsers(title="Available Commands")
experiments.experiment.fill_command_line_parser(SUBPARSERS.add_parser("experiment"))
experiments.pilot_study.fill_command_line_parser(SUBPARSERS.add_parser("pilot"))
experiments.report.fill_command_line_parser(SUBPARSERS.add_parser("report"))
ARGUMENTS = PARSER.parse_args()
ARGUMENTS.func(ARGUMENTS)
