import evcouplings.management.computeJob.ComputeJobInterface as cji


class ComputeJobStdout(cji.ComputeJobInterface):

    def __init__(self, config):
        super(ComputeJobStdout, self).__init__(config)

        # Fallback: if no management section is defined, this will just log current status for job
        self.management = self.config.get("management", {
            "job_name": "Current job",
            "job_group": "unknown"
        })

        self.job_name = self.management.get("job_name", "Current job")
        self.stage = "none"
        self.status = "none"

    def update_job_status(self, status=None, stage=None):
        if stage is not None:
            print("{} is entering stage {}".format(self.job_name, stage))
            self.stage = stage
        elif status is not None:
            print("{} status has changed to {}".format(self.job_name, status))
            self.status = status

    def get_jobs_from_group(self):
        print("This function has no meaning in the context of the local Compute Job.")

    def get_job(self):
        print("{} is in stage '{}' and status '{}'".format(self.job_name, self.stage, self.status))