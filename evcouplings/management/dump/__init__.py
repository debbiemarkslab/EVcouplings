from evcouplings.management.dump.AzureDumper import AzureDumper
from evcouplings.management.dump.LocalDumper import LocalDumper

DUMPERS = {
    "local": LocalDumper,
    "azure": AzureDumper
}


def get_dumper(config):
    dumper = config.get("dumper").get("location")

    if dumper is None:
        dumper = "local"

    return DUMPERS.get(dumper)