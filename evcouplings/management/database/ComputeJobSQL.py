from sqlalchemy import (
    Column, String, DateTime,
    create_engine, func
)
from sqlalchemy.orm import sessionmaker, load_only
from sqlalchemy.ext.declarative import declarative_base

from evcouplings.utils import EStatus
from evcouplings.utils.config import MissingParameterError
import evcouplings.management.database.ComputeJobInterface as cji


_Base = declarative_base()


class _ComputeJob(_Base):
    """
    Single compute job. Holds general information about job
    and its status, but not about individual parameters
    (these are stored in config file to keep table schema
    stable).
    """
    __tablename__ = "runs"

    # human-readable job name (must be unique)
    name = Column(String(100), primary_key=True)

    # location - not used right now, but could
    # be used to point to cloud locations, etc.
    location = Column(String(1024))

    # the job_hash this job is associated with
    # (foreign key in group key if this table
    # is present, e.g. in webserver).
    group_id = Column(String(32))

    # job status ("pending", "running", "finished",
    # "failed", "terminated")
    status = Column(String(50))

    # stage of computational pipeline
    # ("align", "couplings", ...)
    stage = Column(String(50))

    # time the job started running
    time_started = Column(DateTime())

    # time the job finished running
    time_finished = Column(DateTime())


class ComputeJobSQL(cji.ComputeJobInterface):

    def __init__(self, config):
        super(ComputeJobSQL, self).__init__(config)

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
        engine = create_engine(self.database_uri)
        Session = sessionmaker(bind=engine)
        session = Session()

        # make sure all tables are there in database
        _Base.metadata.create_all(bind=engine)

        # see if we can find the job in the database already
        q = session.query(_ComputeJob).filter_by(name=self.job_name)
        num_rows = len(q.all())

        # create new entry if not already existing
        if num_rows == 0:
            r = _ComputeJob(
                name=self.job_name,
                status=EStatus.PEND,
                stage="unknown"
            )
            session.add(r)
            session.commit()
        else:
            # can only be one row due to unique constraint
            r = q.one()

        # if status is given, update
        if status is not None:
            r.status = status

        # if stage is given, update
        if stage is not None:
            r.stage = stage

        # if start time has not been set yet, do so
        if r.time_started is None:
            r.time_started = func.now()

        # update finish time (i.e. final finish
        # time when job status is set for the last time)
        r.time_finished = func.now()

        # commit changes to database
        session.commit()

        # close session again
        session.close()
        return

    def get_job(self):
        # extract database information from job configuration
        # (database URI and job id), as well as job prefix
        uri = self.config.get("database_uri", None)
        name = self.config.get("management",{}).get("job_name")

        # make sure all required fields are defined
        if uri is None or name is None:
            raise MissingParameterError(
                "[DATABASE] Missing 'database_uri' or 'job_name' parameters in incoming config"
            )

        # connect to DB and create session
        engine = create_engine(uri)
        Session = sessionmaker(bind=engine)
        session = Session()

        # make sure all tables are there in database
        _Base.metadata.create_all(bind=engine)

        try:
            # see if we can find the job in the database already
            result = session.query(_ComputeJob)\
                .filter(_ComputeJob.name == name) \
                .options(load_only("name", "prefix", "status", "group_id", "time_started", "time_finished")) \
                .one()

        except:
            session.rollback()
            raise

        finally:
            session.close()

        return result

