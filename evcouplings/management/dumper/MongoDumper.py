import tarfile
import os
from evcouplings.management.dumper.ResultsDumperInterface import ResultsDumperInterface
from evcouplings.utils import valid_file, temp
from pymongo import MongoClient
import gridfs
import re
from copy import deepcopy


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

        # This is used to define a bucket for this job
        # https://docs.mongodb.com/manual/reference/limits/#restrictions-on-db-names
        self._nice_job_name = re.sub(r'[/|\\. "$*<>:?]', "_", self._job_name)

        assert len(self._nice_job_name) < 64, \
            "This job name's length is too long. You must shorten it. Name: {}".format(self._nice_job_name)

        self._archive = self._management.get("archive")
        self._tracked_files = self._dumper.get("tracked_files")

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

        index = self.write_file(tar_file, aliases=['results_archive'])

        return index

    def tar_path(self):
        client = MongoClient(self._database_uri)
        db = client[self._nice_job_name]
        fs = gridfs.GridFS(db)

        results_archive = fs.find_one(filter={
            "job_name": self._job_name,
            "aliases": "results_archive"
        })

        return results_archive._id

    def download_tar(self):
        index = self.tar_path()

        client = MongoClient(self._database_uri)
        db = client[self._nice_job_name]
        fs = gridfs.GridFS(db)

        results_archive = fs.get(index)

        temp_file = temp()
        f = open(temp_file, 'wb')
        f.write(results_archive.read())

        return temp_file

    def write_file(self, file_path, aliases=None):
        assert file_path is not None, "You must pass the location of a file"

        _, upload_name = os.path.split(file_path)

        client = MongoClient(self._database_uri)
        db = client[self._nice_job_name]
        fs = gridfs.GridFS(db)

        with open(file_path, "rb") as f:
            if aliases is not None:
                index = fs.put(f, job_name=self._job_name, filename=upload_name, aliases=aliases)
            else:
                index = fs.put(f, job_name=self._job_name, filename=upload_name)

        client.close()

        return index

    def write_files(self):
        # TODO: Write each single file to blob in correct folder structure
        pass

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
                        index = self.write_file(f, aliases=[k, f])
                        intermediate_result.append(index)
                result[k] = intermediate_result
            else:
                if valid_file(out_config[k]):
                    index = self.write_file(out_config[k], aliases=[k])
                    result[k] = index

        return result

    def clear(self):
        client = MongoClient(self._database_uri)
        return client.drop_database(self._nice_job_name)


