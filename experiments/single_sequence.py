"""
experiments.single_sequence
===========================

Functionality for running a single sequence.
"""

import glob
import os.path
import subprocess
import time
from typing import List
import numpy
import tracking.tmft
import modules.utils
import utilities.loss_data


class Sequence:
    """
    Metadata for the single sequence.

    :param List[tracking.tmft.BoundingBox] ground_truth_data: The list ground truth bounding boxes
        for this sequence.
    :param List[str] image_files: The list of paths to the sequence image files.
    :raises ValueError: if there unequal numbers of ground truth bounding boxes and image files.

    .. py:attribute:: name
        :type: str

        The name of the sequence.

    .. py:attribute:: ground_truth_data
        :type: List[tracking.tmft.BoundingBox]

        The list of ground truth axis-aligned bounding boxes.

    .. py:attribute:: image_files
        :type: List[str]

        The list of image files in the sequence.

    """

    def __init__(self, name, ground_truth_data, image_files):
        if len(ground_truth_data) != len(image_files):
            raise ValueError(
                f"{len(ground_truth_data)} ground truth data does not match {len(image_files)} "
                "image files."
            )
        self.name = name
        self.ground_truth_data = ground_truth_data
        self.image_files = image_files


def run_smoke_test(configuration: dict) -> None:
    """
    Run a smoke test on a specific OTB sequence.

    :param dict configuration: The experimental configuration. This must have the "tmft" key. The
        "tmft" key contains the runtime configuration for TMFT.
    """
    if "tmft" not in configuration:
        raise KeyError("The key 'tmft' is required in the configuration.")
    # Force the random seed to 0 so randomness doesn't affect the smoke test results.
    configuration["tmft"]["random_seed"] = 0
    configuration["tmft"]["save_loss"] = True
    sequence = make_sequence(
        "~/Videos/otb/Deer",
        limits=None
        if "frame_count" not in configuration["smoke_test"]
        else (0, configuration["smoke_test"]["frame_count"]),
    )
    boxes, times, loss_data = _track_sequence(configuration["tmft"], sequence)
    if (
        "save_times" in configuration["smoke_test"]
        and configuration["smoke_test"]["save_times"]
    ):
        _save_times(times)
    print("\nCalculating results")
    assert len(boxes) == len(sequence.ground_truth_data[1:])
    ground_truth_data = numpy.array(
        [tracking.tmft.to_numpy_array(t) for t in sequence.ground_truth_data[1:]]
    )
    overlaps = modules.utils.overlap_ratio(
        numpy.array(boxes), ground_truth_data, as_percentage=True
    )
    offsets = modules.utils.center_offsets(
        numpy.array(boxes), ground_truth_data, whole_pixels=True
    )
    try:
        expected_overlaps, expected_offsets = _load_expected_metrics(
            "Deer", ground_truth_data
        )
        _print_table(
            _compare_results(overlaps, expected_overlaps, offsets, expected_offsets),
            ("IoU", "True IoU", "Delta", "Offset", "True Offset", "Delta"),
        )
    except FileNotFoundError:
        print("Could not load expected tracking results. Skipping analysis.")
    # utilities.loss_data.write_training_records(loss_data, "smoke_test/deer_loss.txt")


def run(configuration: dict, sequence_path: str) -> None:
    """
    Run TMFT one a single sequence.

    :param dict configuration: The experimental configuration. This must have the "tmft" key. The
        "tmft" key contains the runtime configuration for TMFT.
    :param str sequence_path: The path to the sequence to run. This cannot be ``None`` or empty. It
        may be a relative path; this routine will sanitize the path.
    :raises KeyError: if "tmft" is not in the supplied ``configuration``.
    :raises TypeError: if ``sequence_path`` is not a string.
    :raises FileNotFoundError: if the file specified by ``sequence_path`` does not exist.
    :raises NotADirectoryError: if ``sequence_path`` points to a file instead of a directory.
    """
    if "tmft" not in configuration:
        raise KeyError("The key 'tmft' is required in the configuration.")
    sequence = make_sequence(sequence_path)
    return _track_sequence(configuration["tmft"], sequence)


def _track_sequence(configuration: dict, sequence: Sequence) -> numpy.array:
    """
    Track the given sequence.

    :param dict configuration: The TMFT configuration.
    :param Sequence sequence: The meta-data for the sequence to track.
    :return: The tracked bounding boxes.
    :rtype: np.array
    """
    tracker = tracking.tmft.Tmft(configuration)
    image = tracking.tmft.open_image(sequence.image_files[0])
    print("Initializing")
    tracker.initialize(image, sequence.ground_truth_data[0])
    boxes = []
    times = []
    for i, image_path in enumerate(sequence.image_files[1:]):
        image = tracking.tmft.open_image(image_path)
        track_time = time.time()
        box = tracker.find_target(image)
        track_time = time.time() - track_time
        boxes.append(box)
        times.append(track_time)
        _progress_bar(i + 1, 0, len(sequence.image_files[1:]), 80, "Tracking")
    return numpy.array(boxes), numpy.array(times), tracker.training_records


def make_sequence(sequence_path: str, limits=None) -> Sequence:
    """
    Analyze the files at the given sequence path, and create the sequence meta-data.

    :param str sequence_path: The path the sequence.
    :return: The meta-data necessary to track the sequence.
    :rtype: SequenceMetadata
    """
    if not isinstance(sequence_path, str):
        raise TypeError("The sequence_path must be a string.")
    sequence_path = os.path.abspath(os.path.expanduser(sequence_path))
    if not os.path.exists(sequence_path):
        raise FileNotFoundError(f"The sequence path '{sequence_path}' does not exist.")
    if os.path.isfile(sequence_path):
        raise NotADirectoryError(
            f"The sequence path '{sequence_path}' is not a directory."
        )
    sequence = Sequence(
        os.path.basename(sequence_path),
        load_ground_truth_data(find_ground_truth_file(sequence_path)),
        find_image_files(sequence_path),
    )
    if isinstance(limits, tuple) and len(limits) > 1:
        sequence.ground_truth_data = sequence.ground_truth_data[limits[0] : limits[1]]
        sequence.image_files = sequence.image_files[limits[0] : limits[1]]
    return sequence


def find_ground_truth_file(sequence_path: str) -> str:
    """
    Find the path to the sequence's ground truth data file.

    :param str sequence_path: The sequence's root path.
    :return: The path to the ground truth data file.
    :rtype: str
    :raises FileNotFoundError: if no ground truth file can be found.
    """
    paths = glob.glob(f"{sequence_path}/**/groundtruth_rect.txt", recursive=True)
    if len(paths) == 1:
        return paths[0]
    paths = glob.glob(f"{sequence_path}/**/groundtruth.txt", recursive=True)
    if len(paths) == 1:
        return paths[0]
    raise FileNotFoundError(
        f"Cannot find a single ground truth file in {sequence_path}"
    )


def load_ground_truth_data(ground_truth_path: str) -> List[tracking.tmft.BoundingBox]:
    """
    Read the ground truth bounding box data from the ground truth file.

    :param str ground_truth_path: The path the sequence's ground truth data.
    :return: A list of axis-aligned bounding boxes.
    :rtype: tracking.tmft.BoundingBox[]
    """
    with open(ground_truth_path, "r") as ground_truth_file:
        lines = ground_truth_file.readlines()
    lines = [line.strip().split(",") for line in lines]
    return parse_bounding_boxes(lines)


def parse_bounding_boxes(lines: List[List[str]]) -> List[tracking.tmft.BoundingBox]:
    """
    Create a list of bounding boxes from a list of string lists.

    :param List[List[str]] lines: The bounding box data to parse. This is a 2D list of strings. Each
        string must be a number. Each row must have four or eight strings. If each row contains four
        strings, the numbers are interpreted as [x, y, width, height]. If each row contains eight
        strings, the numbers are interpreted as four x,y pairs. In this case, the created bounding
        box will circumscribe actual box described by the string data.
    :return: A list of axis-aligned bounding boxes.
    :rtype: List[tracking.tmft.BoundingBox]
    """
    if len(lines[0]) == 8:
        raise NotImplementedError(
            "Creating axis-aligned bounding boxes from non-aligned boxes is not yet supported."
        )
    if len(lines[0]) == 4:
        boxes = [
            tracking.tmft.BoundingBox(
                int(line[0]), int(line[1]), int(line[2]), int(line[3])
            )
            for line in lines
        ]
        return boxes
    raise ValueError(
        f"Bounding box data contains {len(lines[0])} values, but only 4 and 8 are supported."
    )


def find_image_files(sequence_path: str) -> List[str]:
    """
    Find the image files for a sequence.

    :param str sequence_path: The root path for the sequence.
    :return: A list of file paths for all the image file. The list will be sorted in ascending
        order.
    :rtype: List[str]
    :raises FileNotFoundError: if no image files are found
    """
    file_list = glob.glob(f"{sequence_path}/**/*.jpg", recursive=True)
    if not file_list:
        raise FileNotFoundError(f"No image files were found in {sequence_path}.")
    file_list.sort()
    return file_list


def _progress_bar(
    current: int, minimum: int, maximum: int, width: int, label: str
) -> None:
    width = (
        width - 2 - len(label) - 1 - 5
    )  # subtract the [ and ] characters that encompass the bar
    x = int((current - minimum) / (maximum - minimum) * width)
    percent = int((current - minimum) / (maximum - minimum) * 100)
    percent_text = f" {percent:3}% "
    bar = "=" * x + " " * (width - x)
    print(label, " [", bar, "]", percent_text, sep="", end="\r")


def _load_expected_boxes(sequence_name: str, frame_range=None) -> numpy.array:
    if frame_range is None:
        skip_rows = 0
        max_rows = None
    else:
        skip_rows = frame_range[0]
        max_rows = frame_range[1] - frame_range[0]
    with open(f"results/otb/OTB2015/TMFT/{sequence_name}.txt") as input_file:
        return numpy.loadtxt(
            input_file, delimiter=",", skiprows=skip_rows, max_rows=max_rows
        )


def _compare_results(
    actual_overlaps: numpy.array,
    expected_overlaps: numpy.array,
    actual_offsets: numpy.array,
    expected_offsets: numpy.array,
) -> numpy.array:
    """
    Compare actual results with expected results:

    :param numpy.array actual_overlaps: The list of actual IoU data.
    :param numpy.array expected_overlaps: The list of expected IoU data.
    :param numpy.array actual_offsets: The list of actual center offset data.
    :param numpy.array expected_offsets: The list of expected center offset data.
    :returns: A 2D array of result information. Each row represents a single frame from the tracked
        sequence. The columns are actual IoU, expected IoU, delta IoU, actual offset, expected
        offset, delta offset.
    :rtype: np.array
    """
    return numpy.stack(
        (
            actual_overlaps,
            expected_overlaps,
            actual_overlaps - expected_overlaps,
            actual_offsets,
            expected_offsets,
            actual_offsets - expected_offsets,
        ),
        axis=1,
    )


def _print_table(data: numpy.array, column_headers) -> None:
    """
    Print a set of 1D arrays as a table.

    :param numpy.array data: A 2D numpy array with the data to print.
    :parma tuple column_headers: A set of column headers for data. This must be ``None``, or a
        tuple of strings, with the number of strings matching the number of columns in ``data``.
    """
    if isinstance(column_headers, tuple):
        print("\n", "  ".join(column_headers))
        print("-" * 75)
    print(data)


def _load_expected_metrics(
    sequence_name: str, ground_truth_data: numpy.array
) -> numpy.array:
    boxes = _load_expected_boxes(sequence_name, (1, 1 + len(ground_truth_data)))
    overlaps = modules.utils.overlap_ratio(boxes, ground_truth_data, as_percentage=True)
    offsets = modules.utils.center_offsets(boxes, ground_truth_data, whole_pixels=True)
    return overlaps, offsets


def _save_times(times: numpy.array) -> None:
    """
    Save an array of frame times to a text file.

    :param numpy.array times: The frame tracking times to save.
    """
    completion = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        check=True,
        encoding="utf-8",
    )
    with open("smoke_test_times.txt", "a") as times_file:
        numpy.savetxt(
            times_file,
            numpy.reshape(times, (1, -1)),
            delimiter=",",
            fmt="%.2f",
            newline="",
        )
        times_file.write(f"#{str(completion.stdout).strip()}\n")


def _read_times() -> numpy.array:
    """
    Read a table of frame tracking times from a file.

    :return: The data read from the file.
    :rtype: numpy.array
    """
    values = numpy.genfromtxt("smoke_test_times.txt", comments="#", delimiter=",",)
    return values


def _write_boxes_to_file(file_path: str, boxes: numpy.array) -> None:
    """
    Write the set of bounding boxes to a file.

    :param str file_path: The path to the file to write.
    :param numpy.ndarray boxes: The boxes to write.
    """
    with open(file_path, "w") as output_file:
        for box in boxes:
            output_file.write(",".join(box))
            output_file.write("\n")
