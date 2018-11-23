"""
MongoDB-based result tracker

Stores outconfig in document and files referenced in outconfig in GridFS.
Using this tracker requires installing the pymongo package.

# mongod --config /usr/local/etc/mongod.conf

Authors:
  Thomas A. Hopf
"""

from pymongo import MongoClient
import gridfs

from evcouplings.utils.tracker.base import ResultTracker


class MongoDBTracker(ResultTracker):
    """
    Tracks compute job results in a MongoDB backend

    # TODO: add proper exception handling / retries to this
    """

    def __init__(self, connection_string, job_id, prefix, pipeline, file_list, delete_list, config,
                 retry_max_number=None, retry_wait=None):
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

        # initialize GridFS
        self.fs = gridfs.GridFS(self.db)

    def update(self, status=None, message=None, stage=None, results=None):
        """
        # TODO: check if job exists, if not, create
        # TODO: make sure connection is alive
        # TODO: exception handling if problems
        # TODO: use findOneAndUpdate(...)?
        # TODO: think

        # TODO: store with each file: 1) key, 2) index if list, 3) filename
        # TODO: 4) job id
        # TODO: store cluster ID (optional)
        """
        pass

        """
        # TODO: store config if entry is not there

        if stage is not None:
            pass

        if results is not None:
            pass

        print(self.store_file_keys)
        # check if we need to update results
        if results is not None:
            keep_files, non_file_items = self._partition_results(results)

            print("KEEP FILES:", keep_files)
            print("NON-FILE:", non_file_items)

            # TODO: store in database
            # TODO: think about None files...
            stored_file_items = self._store_files(keep_files)

            # create final set up items that will go into database
            update_items = {
                **non_file_items,
                **stored_file_items,
            }
        else:
            update_items = None

        # TODO: store in database

        # store keys (overwrite)
        # store files... (overwrite)
        # update in one go...
        """
        """
        {
          job_id: <ID>,
          job_group: <ID>,
          pipeline: <monomer|complex>,
          status*: {
            status: <EStatus>,
            stage: <Stage>,
            updated_at: <DateTime>
          }
          input: {
            config: <JSON>,
            resources: <dict_of_blobs>
          },
          output: {
            ... output of current stage --> when finished will contain global outcfg with refs to all files & keys (relevant to web).
          }
        }
        """


