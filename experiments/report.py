"""Generate tracking reports."""

import argparse
import datetime
import glob
import json
import os.path
import got10k.experiments
import experiments.command_line as command_line
import experiments.table as table


def _main() -> None:
    """The main function of the report script."""
    arguments = _parse_command_line()
    _print_experiment_reports(arguments.results_dir, arguments.report_dir)
    _print_pilot_study_report(arguments.results_dir, arguments.report_dir)


def _parse_command_line() -> argparse.Namespace:
    """
    Parse the command line and return the command line arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate tracking reports using the GOT-10k tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--report-dir",
        help="The path to write the reports.",
        default=os.path.expanduser("~/repositories/tmft/reports"),
        action=command_line.PathSanitizer,
    )
    parser.add_argument(
        "results_dir",
        help="The path to the experiment results. The child directories must be benchmark-specific "
        "directories, such as 'OTBtb100'.",
        action=command_line.PathSanitizer,
    )
    return parser.parse_args()


def _today_label() -> str:
    """Get the current date, consistently formatted for a cross reference label."""
    return f"{datetime.date.today():%Y%m%d}"


# ==================================================================================================
# Experiment Reports
# ==================================================================================================
def _print_experiment_reports(results_dir: str, report_dir: str) -> None:
    """
    Create reports for all the experiments in the results directory.

    Args:
        results_dir (str): The directory with the experiment results. Each subdirectory must be a
            GOT-10k benchmark, such as 'OTBtb100'.
        report_dir (str): Write the reports to this directory.
    """
    benchmarks = _find_benchmarks(results_dir)
    if not benchmarks:
        return
    overlap_scores = {}
    robustness_scores = {}
    for benchmark in benchmarks:
        _generate_experiment_report(results_dir, report_dir, benchmark)
        benchmark_overlaps, benchmark_robustess = _load_benchmark_overlap_success(
            report_dir, benchmark
        )
        overlap_scores.update(benchmark_overlaps)
        if benchmark_robustess is not None:
            robustness_scores.update(benchmark_robustess)
    data = _make_experiment_data_table(
        overlap_scores, "Overlap Success", _today_label() + "_overlap_success"
    )
    table.write_table(data, os.path.join(report_dir, "experiment_summary.tex"))
    if robustness_scores:
        data = _make_experiment_data_table(
            robustness_scores, "VOT Robustness", _today_label() + "_vot_robustness"
        )
        table.write_table(data, os.path.join(report_dir, "vot_robustness.tex"))


def _make_experiment_data_table(raw_data: dict, caption: str, label: str) -> table.DataTable:
    """
    Create a data table summarizing the experiment results.

    Args:
        raw_data (dict): Create the table from this data. The keys are the row labels for the
            table. The keys of each value are the column labels. The dict must look like this:

            .. code-block:: json

                {
                    "row1": {
                        "column1": 0.0,
                        "column2": 0.0
                    },
                    "row2": {
                        "column1": 0.0,
                        "column2": 0.0
                    }
                }

        caption (str): The caption for the data table output.
        label (str): The cross reference label for the table output.

    Returns:
        table.DataTable: A data in a table ready for output.
    """
    row_labels = list(raw_data.keys())
    column_labels = list(raw_data[row_labels[0]].keys())
    data = table.DataTable(row_labels, column_labels)
    data.row_labels = row_labels
    data.column_labels = column_labels
    data.caption = caption
    data.label = label
    for row_index, row_label in enumerate(data.row_labels):
        for column_index, column_label in enumerate(data.column_labels):
            if column_label in raw_data[row_label]:
                data[row_index, column_index] = raw_data[row_label][column_label]
    return data


def _find_benchmarks(results_dir: str) -> list:
    """
    Find the benchmarks in the experiment results directory.

    Args:
        results_dir (str): The path to the tracking results. The function searches this path for the
            benchmark-specific results.

    Returns:
        list: The list of benchmarks in the experiment results.
    """
    command_line.print_information("Searching", results_dir, "for benchmarks.")
    benchmarks = [
        os.path.basename(benchmark)
        for benchmark in glob.glob(os.path.join(results_dir, "*"))
        if os.path.basename(benchmark)[0:3] in ["OTB", "UAV", "VOT"]
    ]
    if not benchmarks:
        command_line.print_warning("No benchmarks found.")
        return None
    benchmarks.sort()
    command_line.print_information("Found {}".format(", ".join(benchmarks)))
    return benchmarks


def _generate_experiment_report(result_dir: str, report_dir: str, benchmark: str) -> None:
    """
    Generate a report for a benchmark.

    Args:
        result_dir (str): The path to the benchmark results.
        report_dir (str): The path to write the reports.
        benchmark (str): Generate the report for this benchmark. Examples are 'OTBtb100' and
            'VOT2019'.
    """
    command_line.print_information("Generating reports for", benchmark)
    try:
        experiment = _make_experiment(result_dir, report_dir, benchmark)
    except RuntimeError as error:
        command_line.print_warning(str(error))
        return
    trackers = _find_trackers(os.path.join(result_dir, benchmark))
    experiment.report(trackers)


def _make_experiment(result_dir: str, report_dir: str, benchmark: str):
    """
    Make the GOT-10k experiment object that will create the report.

    Args:
        result_dir (str): The path to the experiment results.
        report_dir (str): The path to write the reports.
        benchmark (str): Create the experiment for this benchmark. Examples are 'OTBtb100' and
            'VOT2019'.

    Returns:
        A GOT-10k experiment object for the requested ``benchmark``.
    """
    if benchmark[:3] == "OTB":
        return got10k.experiments.ExperimentOTB(
            os.path.expanduser("~/Videos/otb"),
            version=benchmark[3:],
            result_dir=result_dir,
            report_dir=report_dir,
        )
    if benchmark == "UAV123":
        return got10k.experiments.ExperimentUAV123(
            os.path.expanduser("~/Videos/uav123"),
            result_dir=result_dir,
            report_dir=report_dir,
        )
    if benchmark[:3] == "VOT":
        return got10k.experiments.ExperimentVOT(
            os.path.expanduser("~/Videos/vot/2019"),
            version=int(benchmark[3:]),
            experiments="supervised",
            result_dir=result_dir,
            report_dir=report_dir,
        )
    raise RuntimeError(f"Unknown benchmark {benchmark}.")


def _find_trackers(result_dir: str) -> list:
    """
    Get the trackers available for an experiment benchmark.

    Args:
        results (str): The path to the benchmark results.

    Returns:
        list: The list of trackers found in the benchmark result directory.
    """
    trackers = [os.path.basename(tracker) for tracker in glob.glob(os.path.join(result_dir, "*"))]
    return sorted(trackers)


def _load_benchmark_overlap_success(report_dir: str, benchmark: str) -> tuple:
    """
    Read overlap success data saved by the benchmark report.

    Args:
        report_dir (str): The directory with the reports.
        benchmark (str): The name of the benchmark.

    Returns:
        tuple: A tuple of (dict, dict|``None``). The first dictionary is the overlap success data.
        The second dictionary is the VOT robustness data. If ``benchmark`` is not a VOT benchmark,
        the second element of the tuple is ``None``.
    """
    if benchmark[:3] not in ["OTB", "UAV", "VOT"]:
        raise RuntimeError(f"Unknown benchmark {benchmark}.")
    file_paths = glob.glob(
        os.path.join(report_dir, benchmark, "**", "performance.json"), recursive=True
    )
    if len(file_paths) > 1:
        raise RuntimeError(
            f"Found {len(file_paths)} performance.json files for {benchmark}. I don't know which "
            "to use."
        )
    with open(file_paths[0], "r") as file:
        data = json.load(file)
    if benchmark[:3] in ["OTB", "UAV"]:
        return (
            {
                _benchmark_to_table_entry(benchmark): {
                    tracker: tracker_data["overall"]["success_score"]
                    for tracker, tracker_data in data.items()
                }
            },
            None,
        )
    # VOT
    return (
        {
            _benchmark_to_table_entry(benchmark): {
                tracker: tracker_data["accuracy"] for tracker, tracker_data in data.items()
            }
        },
        {
            _benchmark_to_table_entry(benchmark): {
                tracker: tracker_data["robustness"] for tracker, tracker_data in data.items()
            }
        },
    )


def _benchmark_to_table_entry(benchmark: str) -> str:
    """
    Convert a GOT-10k benchmark label to a string suitable for printing in a report.

    Args:
        benchmark (str): Convert this benchmark.

    Returns:
        str: The benchmark label converted for use in a report.
    """
    if benchmark.startswith("UAV"):
        return benchmark
    if benchmark.startswith("OTB"):
        return f"OTB-{benchmark[5:]}"
    if benchmark.startswith("VOT"):
        return benchmark.replace("VOT", "VOT ")
    raise ValueError(f"Unknown benchmark {benchmark}.")


# ==================================================================================================
# Pilot Study Report
# ==================================================================================================
def _print_pilot_study_report(results_dir: str, report_dir: str) -> None:
    """
    Create the pilot study report as a table.

    Args:
        results_dir (str): The directory containing the pilot study results database.
        report_dir (str): Write the LaTeX file in this directory.
    """
    try:
        results = _load_pilot_study_database(results_dir)
        data_table = _make_pilot_study_data_table(results)
        table.write_table(data_table, os.path.join(report_dir, "pilot_study.tex"))
    except RuntimeError as error:
        command_line.print_warning(str(error))


def _load_pilot_study_database(results_dir: str) -> dict:
    """
    Load the pilot study results from a database.

    Args:
        results_dir (str): The directory containing the pilot study results database.

    Returns:
        dict: The raw data read from the database.

    Raises:
        RuntimeError: The function raises this exception if
        #. The pilot study database does not exist in ``results_dir``, or
        #. The pilot study database is empty.
    """
    results_path = os.path.join(results_dir, "pilot_results.json")
    if not os.path.isfile(results_path):
        raise RuntimeError(f"{results_path} does not exist, or is not a file.")
    with open(results_path, "r") as pilot_results_file:
        pilot_results = json.load(pilot_results_file)
    if not pilot_results:
        raise RuntimeError(f"{results_path} does not contain pilot study data.")
    return pilot_results


def _make_pilot_study_data_table(pilot_results: dict) -> table.DataTable:
    """
    Create a ``table.DataTable`` from the pilot study results.

    Args:
        pilot_results (dict): The raw data read from the pilot study database.

    Returns:
        table.DataTable: The pilot study data in a table, ready for output.
    """
    sequences = set()
    for tracker_data in pilot_results.values():
        sequences.update(set(tracker_data["scores"].keys()))
    data = table.DataTable(sequences, list(pilot_results.keys()))
    data.caption = "Pilot Study Overlap Success"
    data.label = _today_label() + "_pilot_study"
    data.column_labels = list(pilot_results.keys())
    data.row_labels = sorted(list(sequences))
    for row_index, row_label in enumerate(data.row_labels):
        for column_index, column_label in enumerate(data.column_labels):
            data[row_index, column_index] = pilot_results[column_label]["scores"][row_label]
    return data


if __name__ == "__main__":
    _main()
