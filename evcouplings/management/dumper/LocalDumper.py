import tarfile
import os
import evcouplings.management.dumper.ResultsDumperInterface as rdi
from evcouplings.utils import valid_file
from shutil import copyfile, rmtree


# From https://docs.microsoft.com/en-us/azure/storage/blobs/storage-python-how-to-use-blob-storage
class LocalDumper(rdi.ResultsDumperInterface):

    def __init__(self, config):
        super(LocalDumper, self).__init__(config)

        # IMPORTANT: fallback for old way things are done
        self.parameters = self.config.get("dumper", {
            "storage_location": self.config.get("prefix"),
            "operating_in_same_location": True
        })

        self.storage_location = self.parameters.get("storage_location")
        assert self.storage_location is not None, "Storage location must be defined to know where files should be stored locally"

        self.archive = self.parameters.get("archive", self.config.get("archive"))

    def write_tar(self):
        assert self.archive is not None, "You must define a list of files to be archived"

        # if no output keys are requested, nothing to do
        if self.archive is None or len(self.archive) == 0:
            return

        if not os.path.exists(self.storage_location):
            os.makedirs(self.storage_location)

        tar_file = self.storage_location + ".tar.gz"

        # create archive
        with tarfile.open(tar_file, "w:gz") as tar:
            # add files based on keys one by one
            for k in self.archive:
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
        return self.storage_location + ".tar.gz"

    def download_tar(self):
        # In the case of a local dumper, this is a null operation
        return self.tar_path()

    def write_file(self, file_path):
        assert file_path is not None, "You must pass the location of a file"

        _, upload_name = os.path.split(file_path)

        copyfile(file_path, self.storage_location + upload_name)

    def write_files(self):
        # TODO: Write each single file to blob in correct folder structure
        pass

    def clear(self):
        rmtree(self.storage_location, ignore_errors=True)


