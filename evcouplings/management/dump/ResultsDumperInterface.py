import abc


class ResultsDumperInterface(object, metaclass=abc.ABCMeta):

    def __init__(self, config):
        self.config = config

    def write_zip(self):
        raise NotImplementedError

