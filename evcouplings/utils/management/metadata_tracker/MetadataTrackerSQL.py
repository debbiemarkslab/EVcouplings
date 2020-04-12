"""
SQL based extension of job tracker:
will update an SQL-like instance with current stage and status of a running job.

Authors:
  Thomas A. Hopf
  Christian Dallago
"""

from sqlalchemy import (
    Column, String, DateTime,
    create_engine
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.declarative import declarative_base

from evcouplings.utils import InvalidParameterError
from evcouplings.utils.management.metadata_tracker.MetadataTrackerInterface import (
    MetadataTrackerInterface, DocumentNotFound, DATABASE_NAME)
import datetime


_Base = declarative_base()


def _serialize(sqlAlchemyObject):
    return {
        "job_name": sqlAlchemyObject.job_name,
        "job_group": sqlAlchemyObject.job_group,
        "created_at": sqlAlchemyObject.created_at,
        "updated_at": sqlAlchemyObject.updated_at,
        "status": sqlAlchemyObject.status,
        "stage": sqlAlchemyObject.stage,
    }


class _ComputeJob(_Base):
    """
    Single compute job. Holds general information about job
    and its status, but not about individual parameters
    (these are stored in config file to keep table schema
    stable).
    """
    __tablename__ = DATABASE_NAME

    # human-readable job name (must be unique)
    job_name = Column(String(100), primary_key=True)

    # the job_hash this job is associated with
    # (foreign key in group key if this table
    # is present, e.g. in webserver).
    job_group = Column(String(32))

    # job status ("pending", "running", "finished",
    # "failed", "terminated")
    status = Column(String(50))

    # stage of computational pipeline
    # ("align", "couplings", ...)
    stage = Column(String(50))

    # time the job started running
    created_at = Column(DateTime, default=datetime.datetime.now)

    # last time job was updated
    updated_at = Column(DateTime, default=datetime.datetime.now)


class MetadataTrackerSQL(MetadataTrackerInterface):

    def job_name(self):
        return self._job_name

    def job_group(self):
        return self._job_group

    def status(self):
        return self._status

    def stage(self):
        return self._stage

    def created_at(self):
        return self._created_at

    def updated_at(self):
        return self._updated_at

    def __init__(self, config):
        super(MetadataTrackerSQL, self).__init__(config)

        # Get things from management
        self._management = self.config.get("management")
        if self._management is None:
            raise InvalidParameterError("You must pass a full config file with a management field")

        self._job_name = self._management.get("job_name")
        if self._job_name is None:
            raise InvalidParameterError("config.management must contain a job_name")

        self._job_group = self._management.get("job_group")

        self._metadata_tracker_uri = self._management.get("metadata_tracker_uri")
        if self._metadata_tracker_uri is None:
            raise InvalidParameterError("metadata_tracker_uri must be defined")

        self._status = "initialized"
        self._stage = "initialized"
        self._created_at = datetime.datetime.now()
        self._updated_at = datetime.datetime.now()

        # connect to DB and create session
        engine = create_engine(self._metadata_tracker_uri, poolclass=NullPool)
        Session = sessionmaker(bind=engine)
        session = Session()

        # make sure all tables are there in database
        _Base.metadata.create_all(bind=engine)

        try:
            # see if we can find the job in the database already
            q = session.query(_ComputeJob).get(self._job_name)

            # create new entry if not already existing
            if q is None:
                q = _ComputeJob(
                    job_name=self._job_name,
                    job_group=self._job_group
                )
                session.add(q)
                session.commit()
            else:
                self._created_at = q.created_at
                self._updated_at = q.updated_at
                self._status = q.status
                self._stage = q.stage
        except:
            session.rollback()
            raise

        finally:
            session.close()

    def update_job_status(self, status=None, stage=None):
        """
        Update job status based on configuration and
        update request by pipeline

        Parameters
        ----------
        status : str, optional (default: None)
            If not None, update job status to this value
        stage : str, optional (default: None)
            If not None, update job stage to this value
        """

        # connect to DB and create session
        engine = create_engine(self._metadata_tracker_uri, poolclass=NullPool)
        Session = sessionmaker(bind=engine)
        session = Session()

        # make sure all tables are there in database
        _Base.metadata.create_all(bind=engine)

        try:
            # Finds one or raises exception
            q = session.query(_ComputeJob).filter(_ComputeJob.job_name == self._job_name).one()

            # if status is given, update
            if status is not None:
                self._status = status
                q.status = self._status

            # if stage is given, update
            if stage is not None:
                self._stage = stage
                q.stage = self._stage

            # update finish time (i.e. final finish
            # time when job status is set for the last time)
            q.updated_at = datetime.datetime.now()

            # commit changes to database
            session.add(q)
            session.commit()

            result = _serialize(q)

        except NoResultFound:
            session.rollback()
            raise DocumentNotFound()

        except:
            session.rollback()
            raise

        finally:
            session.close()

        return result

    @staticmethod
    def get_job(job_id, connection_string):
        # connect to DB and create session
        engine = create_engine(connection_string, poolclass=NullPool)
        Session = sessionmaker(bind=engine)
        session = Session()

        # make sure all tables are there in database
        _Base.metadata.create_all(bind=engine)

        result = session.query(_ComputeJob) \
            .get(job_id)

        session.close()

        return _serialize(result)

    @staticmethod
    def get_jobs_from_group(group_id, connection_string):
        # connect to DB and create session
        engine = create_engine(connection_string, poolclass=NullPool)
        Session = sessionmaker(bind=engine)
        session = Session()

        # make sure all tables are there in database
        _Base.metadata.create_all(bind=engine)

        results = session.query(_ComputeJob) \
            .filter(_ComputeJob.job_group == group_id) \
            .all()

        session.close()

        return [_serialize(r) for r in results]
