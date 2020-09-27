"""
SQL-based result tracker (cannot store actual results, only status).
Using this tracker requires installation of the sqlalchemy package.

Regarding using models from different sources in Flask-SQLAlchemy:
https://stackoverflow.com/questions/28789063/associate-external-class-model-with-flask-sqlalchemy

TODO: Note that this tracker doesn't handle job reruns gracefully yet, because the result field will be
progressively overwritten but not reset when the job is rerun.

Authors:
  Thomas A. Hopf
"""

from contextlib import contextmanager
import json
import os
from copy import deepcopy

from sqlalchemy import (
    Column, Integer, String, DateTime, Text,
    create_engine, func
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import DBAPIError
from sqlalchemy.dialects import mysql

from evcouplings.utils.helpers import retry
from evcouplings.utils.config import InvalidParameterError
from evcouplings.utils.tracker import EStatus
from evcouplings.utils.tracker.base import ResultTracker

# create SQLALchemy declarative base for SQL models
Base = declarative_base()

JOB_TABLE_NAME = "evcouplings_jobs"

# work around 65k limitation for mysql (without introducing max length, which would
# cause issues with postgresql)
# see here: https://github.com/sqlalchemy/sqlalchemy/issues/4443
LongText = Text().with_variant(mysql.LONGTEXT(), "mysql")


class SQLTracker(ResultTracker):
    """
    Tracks compute job results in an SQL backend
    """
    def __init__(self, **kwargs):
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
        super().__init__(**kwargs)

        # for SQL tracker, job ID may not be longer than 255 chars to not interfere with older SQL DBs
        if len(self.job_id) > 255:
            raise InvalidParameterError(
                "Length of job_id for SQL tracker may not exceed 255 characters for database compatibility reasons"
            )

        # create SQLAlchemy engine and session maker to
        # instantiate later sessions
        self._engine = create_engine(self.connection_string)
        self._Session = sessionmaker(bind=self._engine)

        # Make sure all tables are there in database
        Base.metadata.create_all(bind=self._engine)

    @contextmanager
    def session_scope(self):
        """
        Provide a transactional scope around a series of operations.
        Source:  https://docs.sqlalchemy.org/en/latest/orm/session_basics.html
        """
        session = self._Session()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

    def get(self):
        """
        Return the current entry tracked by this tracker.
        Does not attempt to retry if database connection fails.
        """
        with self.session_scope() as session:
            query_res = session.query(
                ComputeJob
            ).filter_by(
                job_id=self.job_id
            ).all()

            q = [
                deepcopy(x.__dict__) for x in query_res
            ]

            if len(q) == 0:
                return None
            if len(q) > 1:
                raise ValueError(
                    "Job ID not unique, found more than one job."
                )
            else:
                return q[0]

    def _retry_query(self, func, session, rollback=True):
        """
        Retry database query until success or maximum number of attempts
        is reached

        Parameters
        ----------
        func : callable
            Query function that will be executed until successful
        session : sqlalchemy.orm.session.Session
            SQLALchemy database session
        rollback : bool, optional (default: True)
            Perform rollback of session before reattempt,
            can be set to False for read-only queries

        Returns
        -------
        Result of func()

        Raises
        ------
        ResourceError
            If execution is not successful within maximum
            number of attempts
        """
        if rollback:
            retry_action = session.rollback
        else:
            retry_action = None

        return retry(
            func,
            self.retry_max_number,
            self.retry_wait,
            exceptions=DBAPIError,
            retry_action=retry_action
        )

    def _execute_update(self, session, q, status=None, message=None, stage=None, results=None):
        """
        Wraps update to SQL database (to allow for retries)

        Parameters
        ----------
        session : sqlalchemy.orm.session.Session
            SQLALchemy database session
        q : sqlalchemy.orm.query.Query
            SQLAlchemy query if a job with self.job_id
            already exists

        For remaining parameters, see update()
        """
        # check if we already have some job
        num_rows = len(q.all())

        # create new entry if not already existing
        if num_rows == 0:
            # Note: do not initialize location here, since this should
            # be either set by outside code upon job creation,
            # or based on current working dir of running job
            r = ComputeJob(
                job_id=self.job_id,
                prefix=self.prefix,
                status=EStatus.INIT,
                config=json.dumps(self.config),
                pipeline=self.pipeline,
                time_created=func.now()
            )
            session.add(r)
        else:
            # can only be one row due to unique constraint
            r = q.one()

        # if status is given, update
        if status is not None:
            r.status = status

            # if we switch into running state, record
            # current time as starting time of actual computation
            if status == EStatus.RUN:
                r.time_started = func.now()

                # pragmatic hack to filling in the location if not
                # already set - can only do this based on current directory
                # inside pipeline runner (i.e. when job is started), since
                # any other code that creates the job entry may operate in a
                # different working directory (e.g. batch submitter in evcouplings app)
                if r.location is None:
                    r.location = os.getcwd()

        # if stage is given, update
        if stage is not None:
            r.stage = stage

        # set termination/fail message
        if message is not None:
            r.message = str(message)

        # update timestamp of last modification
        # (will correspond to finished time at the end)
        r.time_updated = func.now()

        # finally, also update results (stored as json)

        if results is not None:
            # first, extract current state in database to dict
            if r.results is not None:
                current_result_state = json.loads(r.results)
            else:
                current_result_state = {}

            # store everything in database except files that are
            # flagged for deletion on filesystem, since we only
            # store the file paths to these files
            result_update = {
                k: v for (k, v) in results.items() if k not in self.delete_list
            }

            # create result update, make sure update overwrites
            # any pre-existing keys
            new_result_state = {
                **current_result_state,
                **result_update
            }

            # finally, add updated result state to database record
            r.results = json.dumps(new_result_state)

        session.commit()

    def update(self, status=None, message=None, stage=None, results=None):
        with self.session_scope() as session:
            # see if we can find the job in the database already
            q = self._retry_query(
                lambda: session.query(ComputeJob).filter_by(job_id=self.job_id),
                session=session,
                rollback=False
            )

            # then execute actual update
            self._retry_query(
                lambda: self._execute_update(session, q, status, message, stage, results),
                session=session,
                rollback=True
            )


class ComputeJob(Base):
    """
    Single compute job. Holds general information about job
    and its status, but not about individual parameters
    (these are stored in config file to keep table schema
    stable).
    """
    __tablename__ = JOB_TABLE_NAME

    # internal unique ID of this single compute job
    key = Column(Integer, primary_key=True)

    # human-readable job identifier (must be unique)
    job_id = Column(String(255), unique=True)

    # job prefix
    prefix = Column(String(2048))

    # job pipeline (monomer, complex, ...)
    pipeline = Column(String(128))

    # location - e.g., working dir, remote URI, asf
    location = Column(String(2048))

    # job status ("pending", "running", "finished",
    # "failed", "terminated")
    status = Column(String(128))

    # message upon job failure / termination
    # (e.g. exception, termination code, ...)
    message = Column(LongText)

    # job identifier e.g. on compute cluster
    # e.g. if job should be stopped
    runner_id = Column(String(2048))

    # stage of computational pipeline
    # ("align", "couplings", ...)
    stage = Column(String(128))

    # time the job was created
    time_created = Column(DateTime())

    # time the job started running
    time_started = Column(DateTime())

    # time the job finished running; last
    # update corresponds to time job finished
    time_updated = Column(DateTime())

    # configuration of job (stringified JSON)
    config = Column(LongText)

    # Optional MD5 hash of configuration to identify
    # unique job configurations
    fingerprint = Column(String(32))

    # results of job (stringified JSON)
    results = Column(LongText)
