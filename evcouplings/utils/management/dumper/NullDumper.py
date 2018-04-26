from evcouplings.utils.management.dumper import ResultsDumperInterface


class NullDumper(ResultsDumperInterface):

    def __init__(self, config):
        super(NullDumper, self).__init__(config)

    def write_file(self, file_path):
        return None

    def move_out_config_files(self, out_config):
        return None

    def clear(self):
        return None
