import abc


class ComputeJobInterface(object, metaclass=abc.ABCMeta):

    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def update_job_status(self, status=None, stage=None):
        raise NotImplementedError

    @abc.abstractmethod
    def get_jobs_from_group(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_job(self):
        raise NotImplementedError
