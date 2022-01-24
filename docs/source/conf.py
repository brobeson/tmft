"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# pylint: disable=invalid-name

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))


# -- Project information -----------------------------------------------------

project = "TMFT"
copyright = "2020, brobeson"  # pylint: disable=redefined-builtin
author = "brobeson"

# The full version, including alpha/beta/rc tags
release = "1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.plantuml",
    "sphinx.ext.intersphinx",
]
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

add_module_names = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "haiku"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Generate --help documentation -------------------------------------------
# pylint: disable=wrong-import-position
import argparse
import experiments.experiment
import experiments.pilot_study
import experiments.report


def _write_help_file(filename: str, parser: argparse.ArgumentParser):
    with open(os.path.join("generated", filename), "w") as help_file:
        help_file.write(parser.format_help())


os.makedirs("generated", exist_ok=True)
_write_help_file(
    "experiment_help.rst",
    experiments.experiment.fill_command_line_parser(
        argparse.ArgumentParser(prog="python3 -m experiments.experiment")
    ),
)
_write_help_file(
    "pilot_help.rst",
    experiments.pilot_study.fill_command_line_parser(
        argparse.ArgumentParser(prog="python3 -m experiments.pilot_study")
    ),
)
_write_help_file(
    "report_help.rst",
    experiments.report.fill_command_line_parser(
        argparse.ArgumentParser(prog="python3 -m experiments.report")
    ),
)

tmft_parser = argparse.ArgumentParser(prog="tmft.py")
subparsers = tmft_parser.add_subparsers()
_write_help_file(
    "tmft_experiment_help.rst",
    experiments.experiment.fill_command_line_parser(subparsers.add_parser("experiment")),
)
_write_help_file(
    "tmft_pilot_help.rst",
    experiments.pilot_study.fill_command_line_parser(subparsers.add_parser("pilot")),
)
_write_help_file(
    "tmft_report_help.rst",
    experiments.report.fill_command_line_parser(subparsers.add_parser("report")),
)
