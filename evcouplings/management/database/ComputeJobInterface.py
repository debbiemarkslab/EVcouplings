import abc


class ComputeJobInterface(object, metaclass=abc.ABCMeta):

    def __init__(self, config):
        self.config = config

        self.management = self.config.get("management")
        assert self.management is not None, "You must pass a full config file with a management field"

        self.job_database = self.management.get("job_database")
        assert self.job_database is not None, \
            "You must define job_database parameters in the management section of the config!"

        self.database_uri = self.management.get("database_uri")
        assert self.database_uri is not None, "database_uri must be defined"

        self.job_name = self.management.get("job_name")
        assert self.job_name is not None, "config.management must contain a job_name"

        self.group_id = self.management.get("job_group")
        assert self.group_id is not None, "config.management must contain a job_group"

    def update_job_status(self, status=None, stage=None):
        raise NotImplementedError

    def get_jobs_from_group(self):
        raise NotImplementedError

    def get_job(self):
        raise NotImplementedError