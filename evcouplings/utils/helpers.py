"""
Useful Python helpers

Authors:
  Thomas A. Hopf, Benjamin Schubert
"""

from collections import OrderedDict
from itertools import groupby
from operator import itemgetter
import pickle, json, csv, os, shutil
from os import path
import time
import sys
import jinja2

from evcouplings.utils import InvalidParameterError


class PersistentDict(dict):
    """
    Persistent dictionary with an API compatible with shelve and anydbm.

    The dict is kept in memory, so the dictionary operations run as fast as
    a regular dictionary.

    Write to disk is delayed until close or sync (similar to gdbm's fast mode).

    Input file format is automatically discovered.
    Output file format is selectable between pickle, json, and csv.
    All three serialization formats are backed by fast C implementations.

    https://code.activestate.com/recipes/576642/
    """

    def __init__(self, filename, flag='c', mode=None, format='json', *args, **kwds):
        self.flag = flag                    # r=readonly, c=create, or n=new
        self.mode = mode                    # None or an octal triple like 0644
        self.format = format                # 'csv', 'json', or 'pickle'
        self.filename = filename
        if flag != 'n' and os.access(filename, os.R_OK):
            fileobj = open(filename, 'rb' if format == 'pickle' else 'r')
            with fileobj:
                self.load(fileobj)
        dict.__init__(self, *args, **kwds)

    def sync(self):
        """Write dict to disk"""
        if self.flag == 'r':
            return

        # dont sync if empty
        if not len(self):
            return

        filename = self.filename
        tempname = filename + '.tmp'
        fileobj = open(tempname, 'wb' if self.format =='pickle' else 'w')
        try:
            self.dump(fileobj)
        except Exception:
            os.remove(tempname)
            raise
        finally:
            fileobj.close()
        shutil.move(tempname, self.filename)    # atomic commit
        if self.mode is not None:
            os.chmod(self.filename, self.mode)

    def close(self):
        self.sync()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def dump(self, fileobj):
        # if self is empty do not write to file
        if not self:
            return

        if self.format == 'csv':
            csv.writer(fileobj).writerows(self.items())
        elif self.format == 'json':
            json.dump(self, fileobj, separators=(',', ':'))
        elif self.format == 'pickle':
            pickle.dump(dict(self), fileobj, 2)
        else:
            raise NotImplementedError('Unknown format: ' + repr(self.format))

    def load(self, fileobj):
        # try formats from most restrictive to least restrictive
        for loader in (pickle.load, json.load, csv.reader):
            fileobj.seek(0)
            try:
                return self.update(loader(fileobj))
            except Exception:
                pass
        raise ValueError('File not in a supported format')


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

    Function assumes that start < end for a and b

    .. note::

        Ends of range are not inclusive

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
    if a[0] >= a[1]:
        raise InvalidParameterError("Start has to be smaller than end a[0] < a[1]")
    if b[0] >= b[1]:
        raise InvalidParameterError("Start has to be smaller than end b[0] < b[1]")
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def find_segments(data):
    """
    Find consecutive number segments, based on Python 2.7 itertools recipe

    Parameters
    ----------
    data : iterable
        Iterable in which to look for consecutive number segments (has to be in order)
    """
    segments = []
    for k, g in groupby(enumerate(data), lambda x: x[0] - x[1]):
        cur_segment = list(map(itemgetter(1), g))
        segments.append(
            (cur_segment[0], cur_segment[-1])
        )

    return segments


def render_template(template_file, mapping):
    """
    Render a template using jinja2 and substitute
    values from mapping

    Parameters
    ----------
    template_file : str
        Path to jinja2 template
    mapping : dict
        Mapping used to substitute values
        in the template

    Returns
    -------
    str
        Rendered template
    """
    template_dir, filename = path.split(template_file)

    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )

    template = jinja_env.get_template(filename)

    return template.render(mapping)


class Progressbar(object):
    """
    Progress bar for command line programs

    Parameters
    ----------
    total_size : int
        The total size of the iteration
    bar_length : int
        The visual bar length that gets printed on stdout
    """

    def __init__(self, total_size, bar_length=60):
        self.total_size = total_size
        self.current_size = 0
        self.bar_length = bar_length

    def __iadd__(self, chunk):
        """
        Convenience function of self.update

        Parameters
        ----------
        chunk : int
            The size of the elements that are processed in the current iteration
        """
        self.update(chunk)
        return self

    def update(self, chunk):
        """
        Updates and prints the progress of the progressbar

        Parameters
        ----------
        chunk : int
            The size of the elements that are processed in the current iteration
        """

        self.current_size += chunk
        if self.current_size < self.total_size:
            filled_len = int(round(self.bar_length * self.current_size / float(self.total_size)))
            percents = round(100.0 * self.current_size / float(self.total_size), 1)
            bar = '=' * filled_len + '-' * (self.bar_length - filled_len)
            sys.stdout.write('[%s] %s%s|%s/%s ...\r' % (bar, percents, '%', self.current_size, self.total_size))
            sys.stdout.flush()
        else:
            filled_len = int(self.bar_length)
            bar = '=' * filled_len
            sys.stdout.write('[%s] %s%s|%s/%s ...\r' % (bar, 100.0, '%', self.total_size, self.total_size))
            sys.stdout.flush()
            sys.stdout.write("\n")


def retry(func, retry_max_number=None, retry_wait=None, exceptions=None,
          retry_action=None, fail_action=None):
    """
    Retry to execute a function as often as requested

    Parameters
    ----------
    func : callable
        Function to be executed until succcessful
    retry_max_number : int, optional (default: None)
        Maximum number of retries. If None, will retry forever.
    retry_wait : int, optional (default: None)
        Number of seconds to wait before attempting retry
    exceptions : exception or tuple(exception)
        Single or tuple of exceptions to catch for retrying
        (any other exception will cause immediate fail)
    retry_action : callable
        Function to execute upon a retry
    fail_action
        Function to execute upon final failure
    """
    # initialize maximum number of tries (if None, try forever)
    num_retries = 0

    while True:
        try:
            return func()
        except exceptions:
            # check if we have exhausted the maximum number of retries,
            # if so, fail with the original exception but perform
            # cleanup before
            if retry_max_number is not None and num_retries >= retry_max_number:
                if fail_action is not None:
                    fail_action()

                raise

            # if waiting time is requested, wait before trying again
            if retry_wait is not None:
                time.sleep(retry_wait)

                # execute action before retrying if necessary
                if retry_action is not None:
                    retry_action()

            num_retries += 1
