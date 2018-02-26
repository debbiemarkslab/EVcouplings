import evcouplings.management.dump.ResultsDumperInterface as rdi
from azure.storage.blob import BlockBlobService

# From https://docs.microsoft.com/en-us/azure/storage/blobs/storage-python-how-to-use-blob-storage

class AzureDumper(rdi.ResultsDumperInterface):

    def __init__(self, config):
        super(AzureDumper, self).__init__(config)

        self.parameters = config.dumper

        self.block_blob_service = BlockBlobService(account_name=self.parameters.account_name,
                                                   account_key=self.parameters.account_key)

    def write_zip(self):
        pass

    def write_files(self):
        pass