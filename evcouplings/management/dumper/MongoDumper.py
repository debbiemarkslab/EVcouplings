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

    def write_tar(self, global_state):
        if self._archive is None or len(self._archive) == 0:
            return

        index = self.tar_path()

        if index is not None:
            return index

        tar_file = temp()

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

        if results_archive is not None:
            index = results_archive._id
        else:
            index = None

        client.close()

        return index

    def download_tar(self):
        index = self.tar_path()

        client = MongoClient(self._database_uri)
        db = client[self._nice_job_name]
        fs = gridfs.GridFS(db)

        results_archive = fs.get(index)

        temp_file = temp()
        f = open(temp_file, 'wb')
        f.write(results_archive.read())

        client.close()

        return temp_file

    def write_file(self, file_path, aliases=None):
        assert file_path is not None, "You must pass the location of a file"

        _, upload_name = os.path.split(file_path)

        client = MongoClient(self._database_uri)
        db = client[self._nice_job_name]
        fs = gridfs.GridFS(db)

        db_file = fs.find_one(filter={
            "filename": upload_name,
            "job_name": self._job_name,
        })

        if db_file is not None:
            return db_file._id

        with open(file_path, "rb") as f:
            if aliases is not None:
                index = fs.put(f, job_name=self._job_name, filename=upload_name, aliases=aliases)
            else:
                index = fs.put(f, job_name=self._job_name, filename=upload_name)

        client.close()

        return index

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
        result = client.drop_database(self._nice_job_name)
        client.close()
        return result

    # Particular methods of this implementation
    def get_files(self, alias):
        """
        Find a file or a list of files based on their alias.
        :param alias: can be something like "remapped_pdb_files", the original file path or file name
        :return: A list of ObjectId's or an empty list, if no file is matched
        """
        client = MongoClient(self._database_uri)
        db = client[self._nice_job_name]
        fs = gridfs.GridFS(db)

        files = fs.find(filter={
            "job_name": self._job_name,
            "$or": [{
                "aliases": alias,
            }, {
                "filename": alias
            }]
        })

        files = list(files)

        client.close()

        return files

    @staticmethod
    def serialize_file_list(files):
        """
        Serializes GridOut objects into dictionaries
        :param files: an array of GridOut objects
        :return: and array of dictionaries containing file metadata
        """
        return [{
            "_id": f._id,
            "filename": f.filename,
            "created_at": f.upload_date,
            "aliases": f.aliases
                 } for f in files]

    def get_bucket(self):
        """
        Returns FS bucket for this job. This is the most general approach for storing and retrieving

        :return: the collection (to find on) and the connection (to close, if not used anymore)
        """
        client = MongoClient(self._database_uri)
        db = client[self._nice_job_name]
        fs = gridfs.GridFS(db)

        return fs, client