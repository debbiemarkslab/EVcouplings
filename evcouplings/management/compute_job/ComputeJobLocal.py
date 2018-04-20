from evcouplings.management.compute_job.ComputeJobInterface import ComputeJobInterface
import datetime


def _serialize(computeJobLocalObject):
    return {
        "job_name": computeJobLocalObject.job_name(),
        "job_group": computeJobLocalObject.job_group(),
        "created_at": computeJobLocalObject.created_at(),
        "updated_at": computeJobLocalObject.updated_at(),
        "status": computeJobLocalObject.status(),
        "stage": computeJobLocalObject.stage(),
    }


class ComputeJobLocal(ComputeJobInterface):

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
        super(ComputeJobLocal, self).__init__(config)

        # Fallback: if no management section is defined, this will just log current status for job
        self._management = self.config.get("management", {
            "job_name": "Current job",
            "job_group": "none",
        })

        self._job_name = self._management.get("job_name", "Current job")
        self._job_group = self._management.get("job_group", "none")

        self._status = "init"
        self._stage = "init"
        self._created_at = datetime.datetime.now()
        self._updated_at = datetime.datetime.now()

    def update_job_status(self, status=None, stage=None):
        if stage is not None:
            self._stage = stage
        elif status is not None:
            self._status = status

        self._updated_at = datetime.datetime.now()

        return _serialize(self)

    @staticmethod
    def get_jobs_from_group(group_id, _):
        return [{
            "job_name": None,
            "job_group": group_id,
            "created_at": None,
            "updated_at": None,
            "status": None,
            "stage": None,
        }]

    @staticmethod
    def get_job(job_id, _):
        return {
            "job_name": job_id,
            "job_group": None,
            "created_at": None,
            "updated_at": None,
            "status": None,
            "stage": None,
        }