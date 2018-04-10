from evcouplings.management.compute_job.ComputeJobInterface import ComputeJobInterface
import datetime


#TODO: when logging is implemented, rewrite this:
def log(item):
    pass


class ComputeJobStdout(ComputeJobInterface):

    def __init__(self, config):
        super(ComputeJobStdout, self).__init__(config)

        # Fallback: if no management section is defined, this will just log current status for job
        self.management = self.config.get("management", {
            "job_name": "Current job"
        })

        self.job_name = self.management.get("job_name", "Current job")

        self.status = "init"
        self.stage = "init"
        self.created_at = datetime.datetime.now
        self.updated_at = datetime.datetime.now

    def update_job_status(self, status=None, stage=None):
        if stage is not None:
            log("{} is entering stage {}".format(self.name, stage))
            self.stage = stage
        elif status is not None:
            log("{} status has changed to {}".format(self.name, status))
            self.status = status

        self.updated_at = datetime.datetime.now()

    def get_jobs_from_group(self):
        log("This function has no meaning in the context of the local Compute Job.")

    def get_job(self):
        log("{} is in stage '{}' and status '{}'".format(self.job_name, self.stage, self.status))