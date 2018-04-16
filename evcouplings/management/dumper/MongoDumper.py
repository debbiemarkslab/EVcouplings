import tarfile
import os
from evcouplings.management.dumper.ResultsDumperInterface import ResultsDumperInterface
from evcouplings.utils import valid_file, temp
from shutil import copyfile, rmtree
from pymongo import MongoClient
import gridfs


class MongoDumper(ResultsDumperInterface):

    def __init__(self, config):
        super(MongoDumper, self).__init__(config)

        # Get things from management
        self._management = self.config.get("management")
        assert self._management is not None, "You must pass a full config file with a management field"

        self._job_name = self._management.get("job_name")
        assert self._job_name is not None, "config.management must contain a job_name"

        # Get things from management.dumper (this is where connection string + approach live)
        self._dumper = self._management.get("dumper")
        assert self._dumper is not None, \
            "You must define dumper parameters in the management section of the config!"

        self._database_uri = self._dumper.get("database_uri")
        assert self._database_uri is not None, "database_uri must be defined"

        self._archive = self._management.get("archive")
        self.tracked_files = self._dumper.get("tracked_files")

    def write_tar(self):
        assert self._archive is not None, "You must define a list of files to be archived"

        # if no output keys are requested, nothing to do
        if self._archive is None or len(self._archive) == 0:
            return

        tar_file = temp()

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

        client = MongoClient(self._database_uri)
        db = client.gridfs_runfiles
        fs = gridfs.GridFS(db)

        with open(tar_file, "rb") as f:
            index = fs.put(f, job_name=self._job_name)

        client.close()

        return index

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


