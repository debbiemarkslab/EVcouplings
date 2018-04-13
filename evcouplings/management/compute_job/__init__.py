from evcouplings.management.compute_job.ComputeJobMongo import ComputeJobMongo
from evcouplings.management.compute_job.ComputeJobSQL import ComputeJobSQL
from evcouplings.management.compute_job.ComputeJobLocal import ComputeJobLocal

EStatus = (lambda **enums: type('Enum', (), enums))(
    INIT="initialized",
    PEND="pending",
    RUN="running",
    DONE="done",
    FAIL="failed",  # job failed due to bug
    TERM="terminated",  # job was terminated externally
)

COMPUTE_JOB_TRACKER = {
    "local": ComputeJobLocal,
    "sql": ComputeJobSQL,
    "mongo": ComputeJobMongo
}

def get_compute_job_tracker(config):
    # Fallback mechanism: if management not defined, or if compute_job in management not defined: use local
    compute_job = config\
        .get("management", {})\
        .get("compute_job", {})\
        .get("type", "local")

    return COMPUTE_JOB_TRACKER.get(compute_job)(config)
