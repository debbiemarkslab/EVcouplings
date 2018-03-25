from evcouplings.management.dumper.AzureDumper import AzureDumper
from evcouplings.management.dumper.LocalDumper import LocalDumper

"""
Dumper types. Default and fallback is "local"
"""
DUMPERS = {
    "local": LocalDumper,
    "azure": AzureDumper
}


def get_dumper(config):
    """
    Based on config, get back dumper (where to store zip file and/or run results and/or intermediate results)
    :param config: flattened config at management stage. Expects `dumper` key or falls back on local.
    :return: Object implementing functions of the ResultsDumperInterface class
    """
    # Fallback mechanism: if management not defined, or location in dumper not defined: use local
    dumper = config\
        .get("dumper", {})\
        .get("location", "local")

    return DUMPERS.get(dumper)(config)
