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

        self.job_name = self.config.get("job_name")
        assert self.job_name is not None, "your config must contain a job_name"

    # def write_tar(self):
    #     raise NotImplementedError
    #
    # def read_tar(self):
    #     raise NotImplementedError
    #
    # def write_files(self):
    #     raise NotImplementedError
    #
    # def write_file(self, file_path):
    #     raise NotImplementedError
    #
    # def clear(self):
    #     raise NotImplementedError
