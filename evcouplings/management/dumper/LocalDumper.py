import tarfile
import os
from evcouplings.management.dumper.ResultsDumperInterface import ResultsDumperInterface
from evcouplings.utils import valid_file
from shutil import copyfile, rmtree


# From https://docs.microsoft.com/en-us/azure/storage/blobs/storage-python-how-to-use-blob-storage
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

        self._archive = self._management.get("archive")
        self.tracked_files = self._dumper.get("tracked_files")

    def write_tar(self):
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
                if k not in self.config or self.config[k] is None:
                    continue

                # distinguish between files and lists of files
                if k.endswith("files"):
                    for f in self.config[k]:
                        if valid_file(f):
                            tar.add(f)
                else:
                    if valid_file(self.config[k]):
                        tar.add(self.config[k])

        return tar_file

    def tar_path(self):
        return self._storage_location + ".tar.gz"

    def download_tar(self):
        # In the case of a local dumper, this is a null operation
        return self.tar_path()

    def write_file(self, file_path):
        assert file_path is not None, "You must pass the location of a file"

        _, upload_name = os.path.split(file_path)

        final_path = os.path.join(self._storage_location, upload_name)

        copyfile(file_path, final_path)

        return final_path

    def move_out_config_files(self, out_config):
        # TODO
        pass

    def clear(self):
        rmtree(self._storage_location, ignore_errors=True)


