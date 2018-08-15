"""
Document based extension of results results_tracker:
will copy tracked file to a bucket/collection in mongo,
which will contain "fs.files" and "fs.chunks" collections for the files.

It will also contain a collection "outconfigs" with one document
containing the key-value pares in the outconfigs (except files).

Authors:
  Christian Dallago
"""

import os
import re
import collections
from evcouplings.utils.management.results_tracker.ResultsTrackerInterface import ResultsTrackerInterface
from evcouplings.utils import valid_file, InvalidParameterError
from copy import deepcopy


def _update_dictionary_recursively(original):
    """
    Function to recursively change keys of a dictionary with new values. Needed to avoid having dots or $ in key string
    as this is not supported as keys in Mongo.
    """

    result = {}

    for k, v in original.items():
        if isinstance(v, collections.Mapping):
            result[re.sub(r'[/|\\. "$*<>:?]', "_", k)] = _update_dictionary_recursively(v)
        else:
            result[re.sub(r'[/|\\. "$*<>:?]', "_", k)] = v

    return result


class ResultsTrackerMongo(ResultsTrackerInterface):

    def __init__(self, config):
        super(ResultsTrackerMongo, self).__init__(config)

        # Get things from management
        self._management = self.config.get("management")
        if self._management is None:
            raise InvalidParameterError("You must pass a full config file with a management field")

        self._job_name = self._management.get("job_name")
        if self._job_name is None:
            raise InvalidParameterError("config.management must contain a job_name")

        self._results_tracker_uri = self._management.get("results_tracker_uri")
        if self._results_tracker_uri is None:
            raise InvalidParameterError("results_tracker_uri must be defined")

        # This is used to define a bucket for this job
        # https://docs.mongodb.com/manual/reference/limits/#restrictions-on-db-names
        self._nice_job_name = re.sub(r'[/|\\. "$*<>:?]', "_", self._job_name)
        if len(self._nice_job_name) > 64:
            raise InvalidParameterError("This job name's length is too long. You must shorten it. Name: {}"
                                        .format(self._nice_job_name))

        self._tracked_files = self._management.get("tracked_files")

    def write_file(self, file_path, aliases=None):
        from pymongo import MongoClient
        import gridfs

        if file_path is None:
            raise InvalidParameterError("You must pass the location of a file")

        _, upload_name = os.path.split(file_path)

        client = MongoClient(self._results_tracker_uri)
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

        # Make sure no keys in result will break mongo key indices (aka: no leading $, no . in string).
        result = _update_dictionary_recursively(result)

        # This will store the new outconfig as an anonymous object in the "metadata" collection of the run's db
        self.write_metadata(result)

        return result

    def clear(self):
        from pymongo import MongoClient

        client = MongoClient(self._results_tracker_uri)
        result = client.drop_database(self._nice_job_name)
        client.close()
        return result

    # Particular methods of this class
    def get_files(self, alias):
        """
        Find a file or a list of files based on their alias.

        Parameters
        ----------
        alias Can be something like "remapped_pdb_files", the original file path or file name

        Returns
        -------
        A list of GridOut objects or an empty list, if no file is matched

        """
        from pymongo import MongoClient
        import gridfs

        client = MongoClient(self._results_tracker_uri)
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

        Parameters
        ----------
        files An array of GridOut objects

        Returns
        -------
        Array of dictionaries containing file metadata

        """
        return [{
            "_id": f._id,
            "filename": f.filename,
            "created_at": f.upload_date,
            "aliases": f.aliases
                 } for f in files]

    def get_bucket(self):
        """
        Returns FS bucket for this job. This is the most general approach for storing and retrieving.

        Returns
        -------
        fs, client: the collection (to find on) and the connection (to close, if not used anymore)

        """
        from pymongo import MongoClient
        import gridfs

        client = MongoClient(self._results_tracker_uri)
        db = client[self._nice_job_name]
        fs = gridfs.GridFS(db)

        return fs, client

    def get_metadata_container(self):
        """
        Returns metadata bucket for this job. This is the most general approach for storing and retrieving metadata.

        Returns
        -------
        collection, client: the collection (to find on) and the connection (to close, if not used anymore)

        """
        from pymongo import MongoClient

        client = MongoClient(self._results_tracker_uri)
        db = client[self._nice_job_name]
        collection = db["metadata"]

        return collection, client

    def write_metadata(self, dictionanry):
        """
        Adds dictionary elements as key-value attributes in "metadata" collection of run

        Returns
        -------

        """
        from pymongo import MongoClient

        client = MongoClient(self._results_tracker_uri)
        db = client[self._nice_job_name]
        collection = db["metadata"]

        collection.insert_one(dictionanry)

        client.close()

        return
