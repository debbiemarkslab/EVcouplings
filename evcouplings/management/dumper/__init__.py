from evcouplings.management.dumper.AzureDumper import AzureDumper
from evcouplings.management.dumper.LocalDumper import LocalDumper

DUMPERS = {
    "local": LocalDumper,
    "azure": AzureDumper
}


def get_dumper(config):
    # Fallback mechanism: if management not defined, or location in dumper not defined: use local
    dumper = config\
        .get("dumper", {})\
        .get("location", "local")

    return DUMPERS.get(dumper)(config)
