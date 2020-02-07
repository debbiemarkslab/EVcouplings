"""
Job tracking in databases or other types of stores

All trackers are returned based on conditional imports to avoid
creating dependencies on all types of trackers, which most users
of the pipeline won't ever use.

Authors:
  Thomas A. Hopf
"""

from copy import deepcopy
from os import environ
from evcouplings.utils.config import InvalidParameterError
from evcouplings.utils.tracker.base import NullTracker

TRACKER_USERNAME_KEY = "EVCOUPLINGS_TRACKER_USERNAME"
TRACKER_PASSWORD_KEY = "EVCOUPLINGS_TRACKER_PASSWORD"

# default: try forever
TRACKER_MAX_NUM_RETRIES = None

# default: wait for 5 minutes, unless specified otherwise
TRACKER_RETRY_WAIT = 60

# enumeration of possible job status values
EStatus = (lambda **enums: type('Enum', (), enums))(
    INIT="initialized",
    PEND="pending",
    RUN="running",
    DONE="done",
    FAIL="failed",  # job failed due to bug
    TERM="terminated",  # job was terminated externally
    BAILOUT="bailout",  # pipeline stopped execution, e.g. due to completely hopeless results
)

FINAL_STATES = {EStatus.DONE, EStatus.TERM, EStatus.FAIL, EStatus.BAILOUT}
FAILURE_STATES = {EStatus.TERM, EStatus.FAIL, EStatus.BAILOUT}


def get_result_tracker(config):
    """
    Create result tracker from configuration

    Parameters
    ----------
    config : dict
        Complete job configuration, including
        "global" and "management" sections.

    Returns
    -------
    evcouplings.utils.tracker.ResultTracker
        Job tracker instance according to config
    """
    # first make copy of config so tracker can't influence
    # job in any way by accident
    config = deepcopy(config)

    management = config.get("management", {})
    tracker_type = management.get("tracker_type")

    # if no tracker selected, return NullTracker right away
    # and don't bother with all parameter setup below
    if tracker_type is None:
        return NullTracker()

    # connection string for database (or the like)
    connection_string = management.get("connection_string")

    # get unique job ID, job prefix and pipeline
    job_id = management.get("job_id", None)
    prefix = config.get("global", {}).get("prefix", None)
    pipeline = config.get("pipeline")

    # list of files that tracker should store
    file_list = management.get("tracker_file_list", None)

    # list of files that pipeline will delete
    delete_list = management.get("delete", [])

    # if we don't have these settings, cannot track job
    if connection_string is None:
        raise InvalidParameterError(
            "Must provide parameter 'connection_string' in management section "
            "of config when using a tracker."
        )

    if job_id is None:
        raise InvalidParameterError(
            "Must provide unique 'job_id' in management section "
            "of config when using a tracker."
        )

    # see if we have authentication information in the
    # environment variables (for careful people...)
    # Default is to authenticate using username/password
    # in URI
    env_tracker_username = environ.get(TRACKER_USERNAME_KEY)
    env_tracker_password = environ.get(TRACKER_PASSWORD_KEY)

    # substitute username/password into connection string
    # (will only have an effect if these are present)
    if connection_string is not None:
        connection_string = connection_string.format(
            username=env_tracker_username,
            password=env_tracker_password
        )

    # retry settings
    retry_max_number = management.get("tracker_max_retries", TRACKER_MAX_NUM_RETRIES)
    retry_wait = management.get("tracker_retry_wait", TRACKER_RETRY_WAIT)

    kwargs = {
        "connection_string": connection_string,
        "job_id": job_id,
        "prefix": prefix,
        "pipeline": pipeline,
        "file_list": file_list,
        "delete_list": delete_list,
        "config": config,
        "retry_max_number": retry_max_number,
        "retry_wait": retry_wait
    }

    # all fields that go into database itself rather than inside config
    # are extracted from config object by now; config param as such only serves
    # as record of the entire configuration and shouldn't be accessed inside
    # tracker to extract any sort of parametrization of the tracker
    if tracker_type == "mongodb":
        from evcouplings.utils.tracker.mongodb import MongoDBTracker
        return MongoDBTracker(**kwargs)
    elif tracker_type == "sql":
        from evcouplings.utils.tracker.sql import SQLTracker
        return SQLTracker(**kwargs)
    else:
        raise InvalidParameterError(
            "Not a valid job result tracker: '{}'. "
            "Valid options are: None, 'sql', 'mongodb'".format(tracker_type)
        )
