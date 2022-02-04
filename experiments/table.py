"""Represents tracking results in a table format."""

import copy
import os.path
import subprocess
import numpy
import pylatex


class FormatSpec:
    """
    Encapsulates options for formatting a :py:class:`DataTable` object when writing it.

    Attributes:
        include_mean (bool): Print a row with the arithmetic mean of each column.
        include_median (bool): Print a row with the median of each column.
        mark_best (bool): Mark the best score in each row, including special rows. This only applies
            to data tables with more than one column.

            .. warning::

                This option does not work well with ``transpose``.

        places (int): The number of places to print after the decimal point.
        pretty_format (bool): After writing a source file, run a format tool on the file. For
            example, after writing the table in a LaTeX file, run ``latexindent`` on the file.

            .. warning::

                This is disabled; it generates too many intermediate garbage files.

        transpose (bool): Tranpose the table output. Print the rows as columns, and print the
            columns as rows.
    """

    def __init__(self):
        self.include_mean = True
        self.include_median = True
        self.mark_best = False
        self.places = 3
        self.pretty_format = False
        self.transpose = False


class DataTable:
    """
    Hold data in a table.

    Args:
        row_labels (list): The list of labels for each row in the table. Do not include
            extra rows such as special rows described in the ``format_spec``.
        column_lables (list): The list of labels for each column in the table. Do not include
            extra columns such as a label of the column of row labels.

    Attributes:
        caption (str): The caption for the table.
        data (numpy.ndarray): A 2D ``numpy.ndarray`` with the numerical data. All the data are
            initially NaN. The output functions ignore NaN data; they are treated as empty cells.
        label (str): A label to use for cross referencing in the output. Not all output formats use
            the label.
        row_labels (list): The list of labels for each row.
        column_labels (list): The list of labels for each column.
        format_spec: A :py:class:`FormatSpec` object describing how to print the data table.
    """

    def __init__(self, row_labels: list, column_labels: list):
        self.data = numpy.full((len(row_labels), len(column_labels)), numpy.nan)
        self.row_labels = row_labels
        self.column_labels = column_labels
        self.format_spec = FormatSpec()
        self.caption = ""
        self.label = ""

    @property
    def shape(self) -> tuple:
        """
        The dimensions of the data table, discounting special rows and columns such as labels.

        Returns:
            tuple: (number of rows, number of columns).
        """
        return self.data.shape

    def __getitem__(self, index) -> numpy.ndarray:
        """
        Access numerical data in the table.

        Args:
            index: The index into the ``data``. Any index supported by numpy is valid.

        Returns:
            numpy.ndarray: The value requested by ``index``.
        """
        return self.data[index]

    def __setitem__(self, index, value) -> None:
        """
        Write numerical data in the table.

        Args:
            index: The index into the ``data``. Any index supported by numpy is valid.
            value: The new data to write into the table. The type must be appropriate for the
                ``index``. For example, if ``index`` is a single integer, then ``value`` must be a
                ``numpy.ndarray`` with shape (1, N), where N is the number of columns in this
                data table.
        """
        self.data[index] = value


def number_of_rows(data: DataTable) -> int:
    """
    Get the number of rows in a data table.

    Args:
        data (DataTable): Get the number of rows for this data table.

    Returns:
        int: The number of rows in the data table ``data``.
    """
    return data.shape[0]


def write_table(data: DataTable, file_path: str) -> None:
    """
    Write a data table to a file.

    The function determines what type of content to write based on the file extension in
    ``file_path``. At this time, the function only supports LaTeX files.

    Args:
        data (DataTable): Write this ``DataTable`` object.
        file_path (str): Write the data table to this file.

    Raises:
        ValueError: The function raises this exception if it cannot determine the content type
            from the file extension.
    """
    data_to_print = _finalize_data_table(data)
    extension = os.path.splitext(file_path)[1]
    if extension == ".tex":
        _write_latex_table(data_to_print, file_path)
    else:
        raise ValueError(f"Unknown table type '{extension}'")


def _finalize_data_table(data: DataTable) -> DataTable:
    """
    Copy a data table and make all change necessary to print the data according to the format
    specification.

    Args:
        data (DataTable): Finalize this DataTable.

    Returns:
        DataTable: A copy of the source ``data``, ready for printing.
    """
    final_data = copy.deepcopy(data)
    if final_data.format_spec.include_mean and number_of_rows(data) > 1:
        final_data.data = numpy.vstack([final_data.data, numpy.nanmean(data.data, axis=0)])
        final_data.row_labels.append("Mean")
    if final_data.format_spec.include_median and number_of_rows(data) > 1:
        final_data.data = numpy.vstack([final_data.data, numpy.nanmedian(data.data, axis=0)])
        final_data.row_labels.append("Median")
    if final_data.format_spec.transpose:
        final_data.data = numpy.transpose(final_data.data)
        final_data.row_labels, final_data.column_labels = (
            final_data.column_labels,
            final_data.row_labels,
        )
    return final_data


# ==================================================================================================
# LaTeX tables
# ==================================================================================================
def _write_latex_table(data: DataTable, file_path: str) -> None:
    """
    Write a data table to a LaTeX file.

    Args:
        data (DataTable): Write this ``DataTable``.
        file_path (str): Write the ``data`` to this file.
    """
    _configure_pylatex()
    table = _make_latex_table(data)
    table.generate_tex(file_path[:-4])
    if data.format_spec.pretty_format:
        subprocess.run(["latexindent", "--overwrite", "--silent", file_path], check=False)


def _configure_pylatex() -> None:
    """Configure a few aspects of the pylatex package."""
    # By default, pylatex does not support the S column type. Add it to the list of available types
    # so the tabular environment works correctly.
    pylatex.table.COLUMN_LETTERS.add("S")

    # By default, pylatex appends '%\n' to every line. Just print a newline.
    pylatex.base_classes.Container.content_separator = "\n"


def _make_latex_table(data: DataTable) -> pylatex.Table:
    """
    Create a `pylatex.Table
    <https://jeltef.github.io/PyLaTeX/current/pylatex/pylatex.table.html#pylatex.table.Table>`_
    object and fill it with the provided ``data``.

    Args:
        data (DataTable): Write this data to the ``pylatex.Table`` object.

    Returns:
        pylatex.Table: The completed table.
    """
    table = pylatex.Table(position="!h")
    with table.create(pylatex.Center()):
        if data.caption:
            if data.label:
                table.add_caption(pylatex.NoEscape(rf"{data.caption}\label{{table:{data.label}}}"))
            else:
                table.add_caption(pylatex.NoEscape(data.caption))
        table_spec = _make_latex_table_spec(data)
        with table.create(pylatex.Tabular(table_spec, booktabs=True)) as tabular:
            _write_latex_column_labels(tabular, data.column_labels)
            for row_index in range(data.shape[0]):
                _write_latex_row(
                    tabular, data.row_labels[row_index], data[row_index], data.format_spec
                )
    return table


def _make_latex_table_spec(data: DataTable) -> str:
    """
    Make the specification for the LaTeX table.

    Args:
        data (DataTable): Create a table spec for this DataTable.

    Returns:
        str: The LaTeX table spec suitable for the given ``data``.
    """
    if data.format_spec.transpose and data.shape[1] > 1:
        if data.format_spec.include_median and data.format_spec.include_mean:
            return "l" + "c" * (data.shape[1] - 2) + "|" + "cc"
        if data.format_spec.include_median or data.format_spec.include_mean:
            return "l" + "c" * (data.shape[1] - 1) + "|" + "c"
    return "l" + "c" * data.shape[1]


def _write_latex_column_labels(table: pylatex.Tabular, labels: list) -> None:
    """
    Write a DataTable's column labels to a LaTeX table.

    Args:
        table (pylatex.Tabular): Write the labels to this LaTeX table object.
        labels (list): Write these column labels.
    """
    labels = [pylatex.NoEscape(f"{{{l}}}") for l in labels]
    labels.insert(0, "")
    table.add_row(labels)
    table.add_hline()


def _write_latex_row(
    table: pylatex.Tabular, label: str, row_data: numpy.ndarray, format_spec: FormatSpec
) -> None:
    """
    Write one row of a DataTable to a LaTeX table.

    Args:
        table (pylatex.Tabular): Write the row to this LaTeX table object.
        label (str): The row label, written to the first column.
        row_data (numpy.ndarray): Write this row data to the ``table``.
        format_spec (FormatSpec): Describe how to format data in this row.
    """
    row = [label]
    row.extend([f"{d:.{format_spec.places}f}" if numpy.isfinite(d) else "" for d in row_data])
    if len(row_data) > 1 and format_spec.mark_best:
        # Add one to account for the row label in row[0].
        index = numpy.argmax(row_data) + 1
        row[index] = pylatex.Command("Best", row[index])
    table.add_row(row)
