import abc


class ComputeJobInterface(object, metaclass=abc.ABCMeta):

    def __init__(self, config):
        self.config = config

    def update_job_status(self, status=None, stage=None):
        raise NotImplementedError

    def get_jobs_from_group(self):
        raise NotImplementedError

    def get_job(self):
        raise NotImplementedError
