from evcouplings.management.compute_job.ComputeJobInterface import ComputeJobInterface
import datetime


#TODO: when logging is implemented, rewrite this:
def log(item):
    pass


class ComputeJobStdout(ComputeJobInterface):

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
        super(ComputeJobStdout, self).__init__(config)

        # Fallback: if no management section is defined, this will just log current status for job
        self._management = self.config.get("management", {
            "job_name": "Current job",
            "job_group": "none",
        })

        self._job_name = self._management.get("job_name", "Current job")
        self._job_group = self._management.get("job_group", "none")

        self._status = "init"
        self._stage = "init"
        self._created_at = datetime.datetime.now
        self._updated_at = datetime.datetime.now

    def update_job_status(self, status=None, stage=None):
        if stage is not None:
            log("{} is entering stage {}".format(self._job_name, stage))
            self._stage = stage
        elif status is not None:
            log("{} status has changed to {}".format(self._job_name, status))
            self._status = status

        self._updated_at = datetime.datetime.now()

    def get_jobs_from_group(self):
        log("This function has no meaning in the context of the local Compute Job.")

    def get_job(self):
        log("{} is in stage '{}' and status '{}'".format(self._job_name, self._stage, self._status))