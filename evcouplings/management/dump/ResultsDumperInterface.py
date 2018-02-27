import abc


class ResultsDumperInterface(object, metaclass=abc.ABCMeta):

    def __init__(self, config):
        self.config = config

    def write_tar(self):
        raise NotImplementedError

    def read_tar(self):
        raise NotImplementedError

    def write_files(self):
        raise NotImplementedError

    def write_file(self, file_path):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError
