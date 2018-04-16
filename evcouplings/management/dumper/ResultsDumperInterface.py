import abc


class ResultsDumperInterface(object, metaclass=abc.ABCMeta):

    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def write_tar(self):
        """
        Writes `.tar.gz` based on files in `archive` list. If archive is None, will pass
        :return: A sting indicating the location of the tar (could be fs or HTTP)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def tar_path(self):
        """
        Get tar path
        :return: A sting indicating the location of the tar (could be fs or HTTP)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def download_tar(self):
        """
        Get tar path on local fs. If remote, fetches tar and makes available in temp folder.
        Is not autogarbaged!
        :return: A sting indicating the location of the tar on fs
        """
        raise NotImplementedError

    @abc.abstractmethod
    def move_out_config_files(self, out_config):
        """
        Writes files listed in out_config and dumper's tracked_files
        If used in management after runs finalized, behaves like `cp`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def write_file(self, file_path):
        """
        Writes a file to the new location.
        :param file_path: A path on fs for a file to be shallow-copied to the new location
        :return: A sting indicating the location of the file (could be fs or HTTP)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def clear(self):
        """
        Deletes all content of dumper. Similar to `rm -rf`. Use carefully.
        """
        raise NotImplementedError
