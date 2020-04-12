"""
Abstract interface to handle files and stage-specific output configurations after a stage has been executed.

Authors:
  Christian Dallago
"""

import abc


class ResultsTrackerInterface(object, metaclass=abc.ABCMeta):

    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def move_out_config_files(self, out_config):
        """
        Writes files listed in out_config and results_tracker's tracked_files.
        Behaves like `cp` if used with local interface
        """
        raise NotImplementedError

    @abc.abstractmethod
    def write_file(self, file_path, aliases=None):
        """
        Writes a file to the new location.
        :param file_path: A path on fs for a file to be shallow-copied to the new location
        :param aliases: File aliases (e.g. alignment_file) to be appended to metadata in databases.
        :return: A sting indicating the location of the file (could be fs or HTTP)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def clear(self):
        """
        Deletes all content of results_tracker. Similar to `rm -rf`. Use carefully.
        """
        raise NotImplementedError
