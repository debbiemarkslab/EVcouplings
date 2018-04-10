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

    # Properties

    @property
    @abc.abstractmethod
    def job_name(self):
        raise NotImplementedError

    @job_name.getter
    @abc.abstractmethod
    def job_name(self):
        raise NotImplementedError

    @job_name.setter
    @abc.abstractmethod
    def job_name(self, value):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def job_group(self):
        raise NotImplementedError

    @job_group.getter
    @abc.abstractmethod
    def job_group(self):
        raise NotImplementedError

    @job_group.setter
    @abc.abstractmethod
    def job_group(self, value):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def status(self):
        raise NotImplementedError

    @status.getter
    @abc.abstractmethod
    def status(self):
        raise NotImplementedError

    @status.setter
    @abc.abstractmethod
    def status(self, value):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def stage(self):
        raise NotImplementedError

    @stage.getter
    @abc.abstractmethod
    def stage(self):
        raise NotImplementedError

    @stage.setter
    @abc.abstractmethod
    def stage(self, value):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def created_at(self):
        raise NotImplementedError

    @created_at.getter
    @abc.abstractmethod
    def created_at(self):
        raise NotImplementedError

    @created_at.setter
    @abc.abstractmethod
    def created_at(self, value):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def updated_at(self):
        raise NotImplementedError

    @updated_at.getter
    @abc.abstractmethod
    def updated_at(self):
        raise NotImplementedError

    @updated_at.setter
    @abc.abstractmethod
    def updated_at(self, value):
        raise NotImplementedError
