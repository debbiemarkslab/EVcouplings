"""
MongoDB-based result tracker

Stores outconfig in document and files referenced in outconfig in GridFS.
Using this tracker requires installing the pymongo package.

TODO: Note that this tracker doesn't handle job reruns gracefully yet, because the result field will be
progressively overwritten but not reset when the job is rerun. Upon rerun, all stale files would have to be
removed from GridFS.

# mongod --config /usr/local/etc/mongod.conf

Authors:
  Thomas A. Hopf
"""

import os
from datetime import datetime
from collections.abc import Mapping

from pymongo import MongoClient, errors
import gridfs

from evcouplings.utils.helpers import retry
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
        Return the current entry tracked by this tracker.
        Does not attempt to retry if database connection fails.
        """
        res = self.jobs.find({"job_id": self.job_id})
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
        return retry(
            func,
            self.retry_max_number,
            self.retry_wait,
            exceptions=CATCH_MONGODB_EXCEPTIONS,
        )

    def _insert_file(self, filename, parent_id):
        """
        Insert file from filesystem into database

        Parameters
        ----------
        filename : str
            Path to file that is to be inserted
        parent_id : bson.ObjectId
            MongoDB identifier of job document this
            file is linked to

        Returns
        -------
        dict
            Dictionary with keys "filename" (original file
            path) and "fs_id" (ObjectId of inserted file
            in GridFS)
        """
        def _insert():
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
        Delete file from GridFS

        Parameters
        ----------
        file_entry : dict
            Dictionary with entries "filename" with full
            file path (not used here) and "fs_id" (ObjectId
            of file that should be deleted). This type
            of dictionary is originally created by _insert_file().
        parent_id : bson.ObjectId
            Identifier of parent job document. Currently
            not used in this function.
        """
        def _delete():
            # may run into problems if database was switched
            # from path-based to GridFS-based file handling
            # so pass if accessing fs_id fails
            try:
                target_file = file_entry["fs_id"]
                self.fs.delete(target_file)
            except TypeError:
                pass

        self._retry_query(
            _delete
        )

    @classmethod
    def _apply_to_files(cls, file_mapping, parent_id, func):
        """
        Apply function to a set of file entries (single file
        or list of files) in a dictionary.

        Parameters
        ----------
        file_mapping : dict
            Dictionary with keys that end in "_file"
            (single file path) or "_files" (list of file
            paths).
        parent_id : bson.ObjectId
            MongoDB identifier of job document this
            file is linked to
        func : callable
            Function to apply to every file in file_mapping
            (should be either _insert_file() or _delete_file())

        Returns
        -------
        file_update : dict
            Updated version of file_mapping in which file paths
            are substituted by dictionary that contain items
            "filename" (original path) and "fs_id"(MongoDB ObjectId
            of inserted file).
        """
        file_update = {}

        for file_key, file_items in file_mapping.items():
            if file_key.endswith("_file") and file_items is not None:
                file_update[file_key] = func(file_items, parent_id)
            elif file_key.endswith("_files") and file_items is not None:
                # files may either be a list of simple filenames (else case, more common)
                # or a mapping of file names to additional annotation (if case)
                if isinstance(file_items, Mapping):
                    file_update[file_key] = [
                        {
                            **func(file, parent_id),
                            "value": value
                        }
                        for file, value in file_items.items()
                    ]
                else:
                    file_update[file_key] = [
                        func(file, parent_id) for file in file_items
                    ]

        return file_update

    def _update_results(self, results, current_state):
        """
        Update results in MongoDB, storing files in GridFS

        Parameters
        ----------
        results : dict
            Result document as returned by pipeline, for input
            in MongoDB
        current_state : dict
            Full current state of database entry (including results
            subdocument)

        Returns
        -------
        update : dict
            Update to "results" subdocument in MongoDB.
            To be inserted in database by calling function.
        """
        # no file list given -> store files as paths
        if self.file_list is None:
            # in this case, store everything but those files that
            # are in deletion list; no need to distinguish between
            # file and non-file entries
            results_update = {
                k: v for k, v in results.items() if k not in self.delete_list
            }

            return results_update

        # Otherwise case, store all files that are selected
        # by file_list in GridFS, and delete any older
        # versions; keep other non-file entries as they are

        # first, partition in files and non files

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

        files_for_storage = {
            k: v for (k, v) in files.items()
            if k in self.file_list
        }

        # determine which file entries were already present
        # and need to be deleted because they are replaced
        # by a newer version
        outdated_files = {
            k: v for (k, v) in current_state.get("results", {}).items()
            if k in files_for_storage
        }

        # insert new files, linked to job database entry (parent)
        file_update = self._apply_to_files(
            file_mapping=files_for_storage,
            parent_id=current_state["_id"],
            func=self._insert_file
        )

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

        else:
            # if there is no status field, means we just created the job, so set to init
            if current_state.get("status") is None:
                update["status"] = EStatus.INIT

        # if stage is given, update
        if stage is not None:
            update["stage"] = stage

        # set termination/fail message
        if message is not None:
            update["message"] = str(message)

        # update job results
        if results is not None:
            results_update = self._update_results(results, current_state)

            # merge with other parts of update from above
            update = {
                **update,
                **{
                    ("results." + k): v for (k, v) in results_update.items()
                }
            }

        # second pass of update
        if len(update) > 0:
            def second_query():
                return self.jobs.update_one(
                    {"_id": current_state["_id"]},
                    {"$set": update}
                )

            self._retry_query(second_query)
