"""
Job status database models and handling

Authors:
  Thomas A. Hopf
"""

from sqlalchemy import (
    Column, Integer, String, DateTime,
    create_engine, func
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# enumeration of possible job status values
EStatus = (lambda **enums: type('Enum', (), enums))(
    INIT="initialized",
    PEND="pending",
    RUN="running",
    DONE="done",
    FAIL="failed",  # job failed due to bug
    TERM="terminated",  # job was terminated externally
)

Base = declarative_base()


class ComputeJob(Base):
    """
    Single compute job. Holds general information about job
    and its status, but not about individual parameters
    (these are stored in config file to keep table schema
    stable).
    """
    __tablename__ = "runs"

    # unique ID of this single compute job
    id = Column(Integer, primary_key=True)

    # human-readable job name (must be unique)
    name = Column(String(100), unique=True)

    # job prefix
    prefix = Column(String(1024))

    # path to configuration file used for job
    config_file = Column(String(1024))

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


def update_job_status(config, status=None, stage=None):
    """
    Update job status based on configuration and
    update request by pipeline

    Parameters
    ----------
    config : dict-like
        Job configuration dictionary. Accessed entries
        are {
            global : {prefix}
            management: {database_uri, job_name}
        }
    status : str, optional (default: None)
        If not None, update job status to this value
    stage : str, optional (default: None)
        If not None, update job stage to this value
    """
    # extract database information from job configuration
    # (database URI and job id), as well as job prefix
    mgmt = config.get("management", {})
    uri = mgmt.get("database_uri", None)
    job_name = mgmt.get("job_name", None)
    prefix = config.get("global", {}).get("prefix", None)

    # if we don't have these settings, cannot update job status
    if uri is None or job_name is None:
        return

    # connect to DB and create session
    engine = create_engine(uri)
    Session = sessionmaker(bind=engine)
    session = Session()

    # make sure all tables are there in database
    Base.metadata.create_all(bind=engine)

    # see if we can find the job in the database already
    q = session.query(ComputeJob).filter_by(name=job_name)
    num_rows = len(q.all())

    # create new entry if not already existing
    if num_rows == 0:
        r = ComputeJob(
            name=job_name,
            prefix=prefix,
            status="pending",
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

    # TODO: remove eventually
    y = [x.__dict__ for x in session.query(ComputeJob).filter_by(name=job_name).all()]
    print()
    print(y)
    print()
    # print(y["status"], y["stage"])
    # print(json.dumps(y[0], indent=2))
    # TODO: remove

    # close session again
    session.close()
