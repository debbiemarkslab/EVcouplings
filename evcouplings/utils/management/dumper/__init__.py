from evcouplings.utils.management.dumper import LocalDumper, NullDumper, MongoDumper


"""
Dumper types. Fallback is "null"
"""
DUMPERS = {
    "local": LocalDumper,
    "mongo": MongoDumper,
    None: NullDumper
}


def get_dumper(config):
    """
    Based on config, get back dumper (where to store zip file and/or run results and/or intermediate results)
    :param config: flattened config at management stage. Expects `dumper` key or falls back on local.
    :return: Object implementing functions of the ResultsDumperInterface class
    """
    # Fallback mechanism: if management not defined, or location in dumper not defined: use NullDumper
    dumper = config \
        .get("management", {}) \
        .get("dumper")

    return DUMPERS.get(dumper)(config)
