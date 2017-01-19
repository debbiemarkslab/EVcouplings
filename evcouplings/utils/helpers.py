"""
Useful Python helpers

Authors:
  Thomas A. Hopf
"""

from collections import OrderedDict


class DefaultOrderedDict(OrderedDict):
    """
    Source:
    http://stackoverflow.com/questions/36727877/inheriting-from-defaultddict-and-ordereddict
    Answer by http://stackoverflow.com/users/3555845/daniel

    Maybe this one would be better?
    http://stackoverflow.com/questions/6190331/can-i-do-an-ordered-default-dict-in-python
    """
    def __init__(self, default_factory=None, **kwargs):
        OrderedDict.__init__(self, **kwargs)
        self.default_factory = default_factory

    def __missing__(self, key):
        result = self[key] = self.default_factory()
        return result


def wrap(text, width=80):
    """
    Wraps a string at a fixed width.

    Arguments
    ---------
    text : str
        Text to be wrapped
    width : int
        Line width

    Returns
    -------
    str
        Wrapped string
    """
    return "\n".join(
        [text[i:i + width] for i in range(0, len(text), width)]
    )


def range_overlap(a, b):
    """
    Source: http://stackoverflow.com/questions/2953967/
            built-in-function-for-computing-overlap-in-python
    Note that ends of range are not inclusive

    Parameters
    ----------
    a : tuple(int, int)
        Start and end of first range
        (end of range is not inclusive)
    b : tuple(int, int)
        Start and end of second range
        (end of range is not inclusive)

    Returns
    -------
    int
        Length of overlap between ranges a and b
    """
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))
