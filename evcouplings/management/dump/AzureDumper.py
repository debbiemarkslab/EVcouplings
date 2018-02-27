import tarfile
import os
import evcouplings.management.dump.ResultsDumperInterface as rdi
from azure.storage.blob import BlockBlobService, ContentSettings, PublicAccess
from evcouplings.utils import valid_file, temp


# From https://docs.microsoft.com/en-us/azure/storage/blobs/storage-python-how-to-use-blob-storage
class AzureDumper(rdi.ResultsDumperInterface):

    def __init__(self, config):
        super(AzureDumper, self).__init__(config)

        self.parameters = self.config.get("dumper")
        assert self.parameters is not None, "config.management.dumper must be defined"
        self.job_name = self.config.get("job_name")
        assert self.job_name is not None, "your config must contain a job_name"

        """
        A container name must be a valid DNS name, conforming to the following naming rules:

        1. Container names must start with a letter or number, and can contain only letters, numbers, and the dash (-) character.
        2. Every dash (-) character must be immediately preceded and followed by a letter or number; consecutive dashes are not permitted in container names.
        3. All letters in a container name must be lowercase.
        4. Container names must be from 3 through 63 characters long.
        """

        # The job_name will most likely contain underscores and dots, which should not be in successive order

        self.nice_job_name = self.job_name.replace(".", "-").replace("_", "-")

        assert self.parameters.get("account_name") is not None and self.parameters.get("account_key") is not None,\
            "account_name and account_key must be defined for AzureDumper"
        self.block_blob_service = BlockBlobService(account_name=self.parameters.get("account_name"),
                                                   account_key=self.parameters.get("account_key"))

        # Get files to archive from config.management.dumper.archive.
        # If that's none, get it from config.management.archive.
        # If that's none too, fail.
        self.archive = self.parameters.get("archive", self.config.get("archive"))

    def write_tar(self, tar_file=temp()):
        assert self.archive is not None, "you must define a list of files to be archived"

        # if no output keys are requested, nothing to do
        if self.archive is None or len(self.archive) == 0:
            return

        # create archive
        with tarfile.open(tar_file, "w:gz") as tar:
            # add files based on keys one by one
            for k in self.archive:
                # skip missing keys or ones not defined
                if k not in self.config or self.config[k] is None:
                    continue

                # distinguish between files and lists of files
                if k.endswith("files"):
                    for f in self.config[k]:
                        if valid_file(f):
                            tar.add(f)
                else:
                    if valid_file(self.config[k]):
                        tar.add(self.config[k])

        # Create container if not exists
        self.block_blob_service.create_container(self.job_name, fail_on_exist=False, public_access=PublicAccess.Blob)

        # Copy file
        self.block_blob_service.create_blob_from_path(
            self.nice_job_name,
            self.job_name + ".tar.gz",
            tar_file,
            content_settings=ContentSettings(content_type='application/gzip')
        )

    def read_tar(self):
        temp_file = temp()
        self.block_blob_service.get_blob_to_path(self.nice_job_name, self.job_name + ".tar.gz", temp_file)

        return temp_file

    def write_file(self, file_path):
        assert file_path is not None, "You must pass the location of a file"

        _, upload_name = os.path.split(file_path)

        # Create container if not exists
        self.block_blob_service.create_container(self.nice_job_name, fail_on_exist=False, public_access=PublicAccess.Blob)

        # Copy file
        self.block_blob_service.create_blob_from_path(
            self.nice_job_name,
            upload_name,
            file_path
        )

    def write_files(self):
        assert self.archive is not None, "you must define a list of files to be archived"
        # TODO: Write each single file to blob in correct folder structure
        pass

    def clear(self):

        self.block_blob_service.delete_container(self.nice_job_name)

        pass
