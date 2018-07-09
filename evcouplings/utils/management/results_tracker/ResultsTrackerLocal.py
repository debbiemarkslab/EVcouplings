"""
Local extension of file results_tracker: will copy "tracked_files" to the new location.

Authors:
  Christian Dallago
"""

import os
from copy import deepcopy

from evcouplings.utils.management.results_tracker.ResultsTrackerInterface import ResultsTrackerInterface
from evcouplings.utils import valid_file, InvalidParameterError
from shutil import copyfile, rmtree


class ResultsTrackerLocal(ResultsTrackerInterface):

    def __init__(self, config):
        super(ResultsTrackerLocal, self).__init__(config)

        self._management = self.config.get("management")
        if self._management is None:
            raise InvalidParameterError("You must pass a full config file with a management field")

        self._results_tracker_location = self._management.get("results_tracker_location")
        if self._results_tracker_location is None:
            raise InvalidParameterError("Storage location must be defined to know where files should be stored locally."
                                        " If no storage_location is defined, prefix must be defined.")

        self._job_name = self._management.get("job_name")
        if self._job_name is None:
            raise InvalidParameterError("config.management must contain a job_name")

        self._results_tracker_location = os.path.join(self._results_tracker_location, self._job_name)

        self._operating_in_place = self._results_tracker_location == self.config.get("global").get("prefix")

        self._tracked_files = self._management.get("tracked_files")

    def write_file(self, file_path):

        if self._operating_in_place:
            return file_path

        if file_path is None:
            raise InvalidParameterError("You must pass the location of a file")

        _, upload_name = os.path.split(file_path)

        final_path = os.path.join(self._results_tracker_location, upload_name)

        copyfile(file_path, final_path)

        return final_path

    def move_out_config_files(self, out_config):
        if self._tracked_files is None:
            return out_config

        result = deepcopy(out_config)

        # add files based on keys one by one
        for k in self._tracked_files:
            # skip missing keys or ones not defined
            if k not in out_config or out_config[k] is None:
                continue

            # distinguish between files and lists of files
            if k.endswith("files"):
                intermediate_result = list()
                for f in out_config[k]:
                    if valid_file(f):
                        index = self.write_file(f)
                        intermediate_result.append(index)
                result[k] = intermediate_result
            else:
                if valid_file(out_config[k]):
                    index = self.write_file(out_config[k])
                    result[k] = index

        return result

    def clear(self):
        rmtree(self._results_tracker_location, ignore_errors=True)


