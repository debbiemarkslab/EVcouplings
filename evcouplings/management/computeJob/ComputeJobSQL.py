from sqlalchemy import (
    Column, String, DateTime,
    create_engine
)
from sqlalchemy.orm import sessionmaker, load_only
from sqlalchemy.ext.declarative import declarative_base
import evcouplings.management.computeJob.ComputeJobInterface as cji
import datetime


_Base = declarative_base()


class _ComputeJob(_Base):
    """
    Single compute job. Holds general information about job
    and its status, but not about individual parameters
    (these are stored in config file to keep table schema
    stable).
    """
    __tablename__ = "compute_jobs"

    # human-readable job name (must be unique)
    name = Column(String(100), primary_key=True)

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
    created_at = Column(DateTime(), default=datetime.datetime.now)

    # time the job finished running
    updated_at = Column(DateTime(), default=datetime.datetime.now)


class ComputeJobSQL(cji.ComputeJobInterface):

    def __init__(self, config):
        super(ComputeJobSQL, self).__init__(config)

        # Get things from management
        self.management = self.config.get("management")
        assert self.management is not None, "You must pass a full config file with a management field"

        self.job_name = self.management.get("job_name")
        assert self.job_name is not None, "config.management must contain a job_name"

        self.group_id = self.management.get("job_group")
        assert self.group_id is not None, "config.management must contain a job_group"

        # Get things from management.job_database (this is where connection string + db type live)
        self.compute_job = self.management.get("compute_job")
        assert self.compute_job is not None, \
            "You must define job_database parameters in the management section of the config!"

        self.database_uri = self.compute_job.get("database_uri")
        assert self.database_uri is not None, "database_uri must be defined"

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

        try:
            # see if we can find the job in the database already
            q = session.query(_ComputeJob).get(self.job_name)

            # create new entry if not already existing
            if q is None:
                q = _ComputeJob(
                    name=self.job_name,
                    group_id=self.job_group
                )
                session.commit()

            # if status is given, update
            if status is not None:
                q.status = status

            # if stage is given, update
            if stage is not None:
                q.stage = stage

            # update finish time (i.e. final finish
            # time when job status is set for the last time)
            q.updated_at = datetime.datetime.now

            # commit changes to database
            session.add(q)
            session.commit()
        except:
            session.rollback()
            raise

        finally:
            session.close()

    def get_job(self):
        # connect to DB and create session
        engine = create_engine(self.database_uri)
        Session = sessionmaker(bind=engine)
        session = Session()

        # make sure all tables are there in database
        _Base.metadata.create_all(bind=engine)

        return session.query(_ComputeJob) \
            .filter(_ComputeJob.name == self.job_name) \
            .options(load_only("name", "prefix", "status", "group_id", "time_started", "time_finished")) \
            .one()

    def get_jobs_from_group(self):
        # connect to DB and create session
        engine = create_engine(self.database_uri)
        Session = sessionmaker(bind=engine)
        session = Session()

        # make sure all tables are there in database
        _Base.metadata.create_all(bind=engine)

        return session.query(_ComputeJob) \
            .filter(_ComputeJob.group_id == self.group_id) \
            .options(load_only("name", "prefix", "status", "group_id", "time_started", "time_finished")) \
            .all()