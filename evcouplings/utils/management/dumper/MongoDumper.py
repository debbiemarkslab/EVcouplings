"""
Document based extension of results dumper:
will copy tracked file to a bucket/collection in mongo,
which will contain "fs.files" and "fs.chunks" collections for the files.

It will also contain a collection "outconfigs" with one document
containing the key-value pares in the outconfigs (except files).

Authors:
  Christian Dallago
"""

import os
from evcouplings.utils.management.dumper import ResultsDumperInterface
from evcouplings.utils import valid_file
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

        self._dumper_uri = self._management.get("dumper_uri")
        assert self._dumper_uri is not None, "dumper_uri must be defined"

        # This is used to define a bucket for this job
        # https://docs.mongodb.com/manual/reference/limits/#restrictions-on-db-names
        self._nice_job_name = re.sub(r'[/|\\. "$*<>:?]', "_", self._job_name)

        assert len(self._nice_job_name) < 64, \
            "This job name's length is too long. You must shorten it. Name: {}".format(self._nice_job_name)

        self._tracked_files = self._management.get("tracked_files")

    def write_file(self, file_path, aliases=None):
        assert file_path is not None, "You must pass the location of a file"

        _, upload_name = os.path.split(file_path)

        client = MongoClient(self._dumper_uri)
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
                        index = self.write_file(f, aliases=[k])
                        intermediate_result.append(index)
                result[k] = intermediate_result
            else:
                if valid_file(out_config[k]):
                    index = self.write_file(out_config[k], aliases=[k])
                    result[k] = index

        return result

    def clear(self):
        client = MongoClient(self._dumper_uri)
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
        client = MongoClient(self._dumper_uri)
        db = client[self._nice_job_name]
        fs = gridfs.GridFS(db)

        if alias is None:
            files = fs.find()
        else:
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
        client = MongoClient(self._dumper_uri)
        db = client[self._nice_job_name]
        fs = gridfs.GridFS(db)

        return fs, client