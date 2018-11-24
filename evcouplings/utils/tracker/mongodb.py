"""
MongoDB-based result tracker

Stores outconfig in document and files referenced in outconfig in GridFS.
Using this tracker requires installing the pymongo package.

# mongod --config /usr/local/etc/mongod.conf

Authors:
  Thomas A. Hopf
"""

import os
import time
from datetime import datetime

from pymongo import MongoClient, errors
import gridfs

from evcouplings.utils.system import ResourceError
from evcouplings.utils.tracker.base import ResultTracker
from evcouplings.utils.tracker import EStatus

# exceptions to catch
CATCH_MONGODB_EXCEPTIONS = (
    errors.ConnectionFailure,
    errors.ServerSelectionTimeoutError,
    errors.ExecutionTimeout,
    errors.OperationFailure,
)

JOB_COLLECTION_NAME = "evcouplings_jobs"
FILE_COLLECTION_NAME = "evcouplings_files"


class MongoDBTracker(ResultTracker):
    """
    Tracks compute job results in a MongoDB backend
    """
    def __init__(self, **kwargs):
        """
        Create new MongoDB-based tracker

        Parameters
        ----------
        connection_string : str
            MongoDB connection URI. Must include database name,
            and username/password if authentication is used.
        job_id : str
            Unique job identifier of job which should be tracked
        prefix : str
            Prefix of pipeline job
        pipeline : str
            Name of pipeline that is running
        file_list : list(str)
            List of file item keys from outconfig that should
            be stored in database. If None, will not store files
            in database but store paths to files output by pipeline.
        delete_list : list(str)
            List of file item keys from outconfig that will be deleted
            after run is finished. These files cannot be stored as paths
            to the pipeline result in the output, but they may be stored
            in the MongoDB.
        config : dict(str)
            Entire configuration dictionary of job
        retry_max_number : int, optional (default: None)
            Maximum number of attemps to perform database queries / updates.
            If None, will try forever.
        retry_wait : int, optional (default: None)
            Time in seconds between retries to connect to database
        """
        super().__init__(**kwargs)

        # initialize MongoDB client
        self.client = MongoClient(self.connection_string)

        # get default database (as specified in connection string)
        self.db = self.client.get_database()
        self.jobs = self.db[JOB_COLLECTION_NAME]

        # initialize GridFS
        self.fs = gridfs.GridFS(self.db, collection=FILE_COLLECTION_NAME)

    def get(self):
        """
        Return the current entry tracked by this tracker
        """
        res = self._retry_query(
            lambda: self.jobs.find({"job_id": self.job_id})
        )

        num_documents = res.count()

        if num_documents == 0:
            return None
        if num_documents > 1:
            raise ValueError(
                "Job ID not unique, found more than one job."
            )
        else:
            return res.next()

    def _retry_query(self, func):
        """
        Retry database query until success or maximum number of attempts
        is reached

        Parameters
        ----------
        func : callable
            Query function that will be executed until successful

        Returns
        -------
        Result of func()

        Raises
        ------
        ResourceError
            If execution is not successful within maximum
            number of attempts
        """
        # initialize maximum number of tries (if None, try forever)
        num_retries = 0

        while self.retry_max_number is None or num_retries <= self.retry_max_number:
            print("try...")  # TODO: remove
            try:
                return func()
            except CATCH_MONGODB_EXCEPTIONS as e:
                print("exception", e)  # TODO: remove
                if num_retries >= self.retry_max_number:
                    raise

                # if waiting time is requested, wait before trying again
                if self.retry_wait is not None:
                    time.sleep(self.retry_wait)

                num_retries += 1

        raise ResourceError(
            "Could not successfully execute database query within maximum number of retries"
        )

    def _insert_file(self, filename, parent_id):
        """
        # TODO
        """
        def _insert():
            print("...STORING", filename, "for", parent_id)  # TODO
            with open(filename, "rb") as f:
                return self.fs.put(
                    f,
                    parent_id=parent_id,
                    job_id=self.job_id,
                    filename=filename,
                    time_saved=datetime.utcnow()
                )

        try:
            id_ = self._retry_query(
                _insert
            )
        except OSError as e:
            raise ResourceError(
                "Could not read {} for storing in MongoDB backend".format(
                    filename
                )
            ) from e

        return {"filename": filename, "fs_id": id_}

    def _delete_file(self, file_entry, parent_id):
        """
        # TODO
        """
        def _delete():
            self.fs.delete(file_entry["fs_id"])

        print("-> delete", file_entry)  # TODO: remove
        self._retry_query(
            _delete
        )

    @classmethod
    def _apply_to_files(cls, file_mapping, parent_id, func):
        """
        # TODO
        """
        file_update = {}

        for file_key, file_items in file_mapping.items():
            if file_key.endswith("_file"):
                file_update[file_key] = func(file_items, parent_id)
            elif file_key.endswith("_files"):
                file_update[file_key] = [
                    func(file, parent_id) for file in file_items
                ]

        return file_update

    def _update_files(self, results, current_state):
        """
        # TODO
        """
        # partition in files and non files
        print("--- FILE UPLOAD ---")  # TODO: remove

        # determine which result entries are files
        files = {
            k: v for (k, v) in results.items()
            if k.endswith("_file") or k.endswith("_files")
        }

        # all non-file entries
        other_entries = {
            k: v for (k, v) in results.items() if
            k not in files
        }

        print("other entries:", other_entries)  # TODO

        files_for_storage = {
            k: v for (k, v) in files.items()
            if k in self.file_list
        }
        print("files for storage:", files_for_storage)  # TODO

        # determine which file entries were already present
        # and need to be deleted because they are replaced
        # by a newer version
        outdated_files = {
            k: v for (k, v) in current_state.get("results", {}).items()
            if k in files_for_storage
        }
        print("updated files", outdated_files)

        # insert new files, linked to job database entry (parent)
        file_update = self._apply_to_files(
            file_mapping=files_for_storage,
            parent_id=current_state["_id"],
            func=self._insert_file
        )
        print("FILE UPDATE RESULT", file_update)

        # delete outdated old versions of files
        self._apply_to_files(
            file_mapping=outdated_files,
            parent_id=current_state["_id"],
            func=self._delete_file
        )

        update = {
            **other_entries,
            **file_update
        }

        return update

    def update(self, status=None, message=None, stage=None, results=None):
        # first make sure job is there, need this information for
        # conditional update and file deletion below (cannot do this
        # in a single query unfortunately)
        def first_query():
            return self.jobs.find_one_and_update(
                {"job_id": self.job_id},
                {
                    "$setOnInsert": {
                        "job_id": self.job_id,
                        "prefix": self.prefix,
                        "config": self.config,
                        "pipeline": self.pipeline,
                        "time_created": datetime.utcnow()
                    },
                    "$set": {
                        "time_updated": datetime.utcnow(),
                    }
                },
                upsert=True,
                new=True
            )

        current_state = self._retry_query(first_query)

        # now prepare information for actual update
        update = {}

        # if status is given, update
        if status is not None:
            update["status"] = status

            # if we switch into running state, record
            # current time as starting time of actual computation
            if status == EStatus.RUN:
                update["time_started"] = datetime.utcnow()

                # pragmatic hack to filling in the location if not
                # already set - can only do this based on current directory
                # inside pipeline runner (i.e. when job is started), since
                # any other code that creates the job entry may operate in a
                # different working directory (e.g. batch submitter in evcouplings app)
                if current_state.get("location") is None:
                    update["location"] = os.getcwd()
                    print("Setting location")  # TODO: remove

        else:
            # if there is no status field, means we just created the job, so set to init
            if current_state.get("status") is None:
                update["status"] = EStatus.INIT
                print("Setting init status")  # TODO: remove

        # if stage is given, update
        if stage is not None:
            update["stage"] = stage

        # set termination/fail message
        if message is not None:
            update["message"] = str(message)

        if results is not None:
            # no file list given -> store files as paths
            if self.file_list is None:
                # in this case, store everything but those files that
                # are in deletion list
                print("storing file paths...")  # TODO: remove
                results_update = {
                    k: v for k, v in results.items() if k not in self.delete_list
                }

            else:
                # in this case, store all files that are selected
                # by file_list in GridFS, and delete any older
                # versions
                print("storing files in GridFS")  # TODO: remove
                results_update = self._update_files(results, current_state)

            update = {
                **update,
                **{
                    ("results." + k): v for (k, v) in results_update.items()
                }
            }

        print("ACTUAL UPDATE:", update)

        # second pass of update
        if len(update) > 0:
            def second_query():
                return self.jobs.update_one(
                    {"_id": current_state["_id"]},
                    {"$set": update}
                )

            self._retry_query(second_query)

        print("done...")  # TODO: remove

