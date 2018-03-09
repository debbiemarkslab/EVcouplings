import abc


class ResultsDumperInterface(object, metaclass=abc.ABCMeta):

    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def write_tar(self):
        raise NotImplementedError

    @abc.abstractmethod
    def tar_path(self):
        raise NotImplementedError

    @abc.abstractmethod
    def download_tar(self):
        raise NotImplementedError

    @abc.abstractmethod
    def write_files(self):
        raise NotImplementedError

    @abc.abstractmethod
    def write_file(self, file_path):
        raise NotImplementedError

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError
