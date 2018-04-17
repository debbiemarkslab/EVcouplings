import tarfile
import os
from copy import deepcopy

from evcouplings.management.dumper.ResultsDumperInterface import ResultsDumperInterface
from evcouplings.utils import valid_file
from shutil import copyfile, rmtree


class LocalDumper(ResultsDumperInterface):

    def __init__(self, config):
        super(LocalDumper, self).__init__(config)

        self._management = self.config.get("management")
        assert self._management is not None, "You must pass a full config file with a management field"

        # IMPORTANT: fallback for old way things are done
        self._dumper = self._management.get("dumper", {
            "storage_location": self.config.get("prefix"),
            "operating_in_same_location": True
        })

        self._storage_location = self._dumper.get("storage_location")
        assert self._storage_location is not None, "Storage location must be defined to know " \
                                                   "where files should be stored locally. If no storage_location " \
                                                   "is defined, prefix must be defined."

        self._operating_in_place = self._dumper.get("operating_in_same_location")

        self._archive = self._management.get("archive")
        self._tracked_files = self._dumper.get("tracked_files")

    def write_tar(self, global_state):
        # if no output keys are requested, nothing to do
        if self._archive is None or len(self._archive) == 0:
            return

        if not os.path.exists(self._storage_location):
            os.makedirs(self._storage_location)

        tar_file = self._storage_location + ".tar.gz"

        # create archive
        with tarfile.open(tar_file, "w:gz") as tar:
            # add files based on keys one by one
            for k in self._archive:
                # skip missing keys or ones not defined
                if k not in global_state or global_state[k] is None:
                    continue

                # distinguish between files and lists of files
                if k.endswith("files"):
                    for f in global_state[k]:
                        if valid_file(f):
                            tar.add(f)
                else:
                    if valid_file(global_state[k]):
                        tar.add(global_state[k])

        return tar_file

    def tar_path(self):
        return self._storage_location + ".tar.gz"

    def download_tar(self):
        # In the case of a local dumper, this is a null operation
        return self.tar_path()

    def write_file(self, file_path):

        if self._operating_in_place:
            return file_path

        assert file_path is not None, "You must pass the location of a file"

        _, upload_name = os.path.split(file_path)

        final_path = os.path.join(self._storage_location, upload_name)

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
        rmtree(self._storage_location, ignore_errors=True)


