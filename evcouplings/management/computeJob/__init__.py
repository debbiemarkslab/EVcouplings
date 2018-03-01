from evcouplings.management.computeJob.ComputeJobSQL import ComputeJobSQL
from evcouplings.management.computeJob.ComputeJobStdout import ComputeJobStdout

EStatus = (lambda **enums: type('Enum', (), enums))(
    INIT="initialized",
    PEND="pending",
    RUN="running",
    DONE="done",
    FAIL="failed",  # job failed due to bug
    TERM="terminated",  # job was terminated externally
)

COMPUTEJOBTRACKER = {
    "local": ComputeJobStdout,
    "sql": ComputeJobSQL
}


def get_compute_job_tracker(config):
    # Fallback mechanism: if management not defined, or if compute_job in management not defined: use local
    compute_job = config\
        .get("management", {})\
        .get("compute_job", {})\
        .get("type", "local")

    return COMPUTEJOBTRACKER.get(compute_job)(config)
