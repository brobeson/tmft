"""
utilities.bounding_boxes
========================

Classes and functions related to bounding boxes.
"""

import numpy


def x(box: numpy.array) -> float:
    """
    The X coordinate of a bounding box.

    :param numpy.array box: The box for which to get the X coordinate.
    :return: The X coordinate of a bounding box.
    :rtype: float
    """
    return box[0]


def y(box: numpy.array) -> float:
    """
    The Y coordinate of a bounding box.

    :param numpy.array box: The box for which to get the Y coordinate.
    :return: The Y coordinate of a bounding box.
    :rtype: float
    """
    return box[1]


def width(box: numpy.array) -> float:
    """
    The width of a bounding box.

    :param numpy.array box: The box for which to get the width.
    :return: The width of a bounding box.
    :rtype: float
    """
    return box[2]


def height(box: numpy.array) -> float:
    """
    The height of a bounding box.

    :param numpy.array box: The box for which to get the height.
    :return: The height of a bounding box.
    :rtype: float
    """
    return box[3]


def dimensions(box: numpy.array) -> numpy.array:
    """
    Get the dimensions of the box as a numpy array.

    :param numpy.array box: The box for which to get the dimensions.
    :return: The width and height in a numpy array.
    :rtype: numpy.array
    """
    return box[2:3]


def center(box: numpy.array) -> numpy.array:
    """
    Calculate the center point of a bounding box.

    :param numpy.array box: The box for which to calculate the center.
    :returns: The center point of the box.
    :rtype: numpy.array
    """
    return numpy.array([x(box), +width(box) / 2.0, y(box) + height(box) / 2.0])


def centered_box(box: numpy.array) -> numpy.array:
    """
    Return a bounding box, with the (X,Y) coordinate specifying the center instead of the upper
    left corner.

    :param numpy.array box: The box to move, so to speak.
    :returns: The same box, but with the (X,Y) coordinate in the center.
    :rtype: numpy.array
    """
    return numpy.array(
        [x(box), +width(box) / 2.0, y(box) + height(box) / 2.0, width(box), height(box)]
    )
