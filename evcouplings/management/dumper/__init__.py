from evcouplings.management.dumper.AzureDumper import AzureDumper
from evcouplings.management.dumper.LocalDumper import LocalDumper

DUMPERS = {
    "local": LocalDumper,
    "azure": AzureDumper
}


def get_dumper(config):
    dumper = config.get("dumper").get("location")

    if dumper is None:
        dumper = "local"

    return DUMPERS.get(dumper)(config)
