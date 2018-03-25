import abc


class ComputeJobInterface(object, metaclass=abc.ABCMeta):

    def __init__(self, config):
        """
        Initializer accepts global job config (non-flattened)
        :param config: complete job config
        """
        self.config = config

        pass

    @abc.abstractproperty
    def name(self):
        """
         #TODO: figure out this, apply for:

            "name": self.job_name,
            "job_group": "none",
            "status": self.status,
            "stage": self.stage,
            "created_at": None,
            "updated_at": None

        """
        pass

    @abc.abstractmethod
    def update_job_status(self, status=None, stage=None):
        """
        Updates the status and/or stage of the job. Status should be EStatus
        :param status: EStatus status. Default is None, which won't update it.
        :param stage: string representing compute stage (e.g.: align, compare,...). Default None, which won't update the field
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_jobs_from_group(self):
        """
        Get all compute jobs from job group.
        This implies that management.compute_job and management.job_group are defined.
        Returns sensible results on database-backed extensions of the abstract class only.
        :return: An array of dictionaries containing fields like name, group_id, etc.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_job(self):
        """
        Get this compute job
        :return: A dict with compute job parameters
        """
        raise NotImplementedError
