from evcouplings.utils.management.dumper.LocalDumper import LocalDumper
from evcouplings.utils.management.dumper.NullDumper import NullDumper
from evcouplings.utils.management.dumper.MongoDumper import MongoDumper


# Dumper types. Fallback is "null"
DUMPERS = {
    "local": LocalDumper,
    "mongo": MongoDumper,
    None: NullDumper
}


def get_dumper(config):
    """
    Based on config, get back dumper (where to store zip file and/or run results and/or intermediate results)
    Will fallback to a NullDumper, which won't do anything.

    Parameters
    ----------
    config A complete config (not flattened!)

    Returns
    -------
    Object extending ResultsDumperInterface

    """

    # Fallback mechanism: if management not defined, or location in dumper not defined: use NullDumper
    dumper = config \
        .get("management", {}) \
        .get("dumper")

    return DUMPERS.get(dumper)(config)
