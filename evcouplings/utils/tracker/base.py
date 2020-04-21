"""
Base class to track job results

Authors:
  Thomas A. Hopf
"""

from abc import ABC, abstractmethod

DEFAULT_RESULT_COLLECTION = "evcouplings_jobs"
DEFAULT_FILE_COLLECTION = "evcouplings_files"


class ResultTracker(ABC):
    """
    Track job status and results in a database
    """
    def __init__(self, connection_string, job_id, prefix, pipeline, file_list, delete_list, config,
                 retry_max_number=None, retry_wait=None):
        """
        Create new SQL-based tracker. For now, this tracker will ignore file_list
        and store all file paths in the database except for those in delete_list.

        Parameters
        ----------
        connection_string : str
            SQLite connection URI. Must include database name,
            and username/password if authentication is used.
        job_id : str
            Unique job identifier of job which should be tracked
        prefix : str
            Prefix of pipeline job
        pipeline : str
            Name of pipeline that is running
        file_list : list(str)
            List of file item keys from outconfig that should
            be stored in database. For now, this parameter has no
            effect and all file paths will be stored in database.
        delete_list : list(str)
            List of file item keys from outconfig that will be deleted
            after run is finished. These files cannot be stored as paths
            to the pipeline result in the output.
        config : dict(str)
            Entire configuration dictionary of job
        retry_max_number : int, optional (default: None)
            Maximum number of attemps to perform database queries / updates.
            If None, will try forever.
        retry_wait : int, optional (default: None)
            Time in seconds between retries to connect to database
        """
        self.connection_string = connection_string
        self.job_id = job_id
        self.prefix = prefix
        self.pipeline = pipeline
        self.file_list = file_list
        self.delete_list = delete_list
        self.config = config

        # settings for database update/qery retries
        self.retry_max_number = retry_max_number
        self.retry_wait = retry_wait

    @abstractmethod
    def update(self, status=None, message=None, stage=None, results=None):
        """
        Update job status in tracking backend, create entry if not
        already existing

        Parameters
        ----------
        status : str, optional (default: None)
            If not None, update job status to this value
        message : str, optional (default: None)
            Status message when job fails or is terminated
        stage : str, optional (default: None)
            If not None, update job stage to this value
        results : dict, optional (default: None)
            Update to job results from pipeline (will perform shallow merge)
        """
        raise NotImplementedError


class NullTracker:
    """
    Default tracker that doesn't do anything
    (used if no other tracker specified)
    """
    def update(self, status=None, message=None, stage=None, results=None):
        pass
