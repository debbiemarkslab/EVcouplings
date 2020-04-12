from evcouplings.utils.management.metadata_tracker.MetadataTrackerMongo import MetadataTrackerMongo
from evcouplings.utils.management.metadata_tracker.MetadataTrackerSQL import MetadataTrackerSQL
from evcouplings.utils.management.metadata_tracker.MetadataTrackerLocal import MetadataTrackerLocal


# Status of job. Resolves to string
EStatus = (lambda **enums: type('Enum', (), enums))(
    INIT="initialized",
    PEND="pending",
    RUN="running",
    DONE="done",
    FAIL="failed",  # job failed due to bug
    TERM="terminated",  # job was terminated externally
)

# Tracker types. Default and fallback is "local"
METADATA_TRACKER = {
    "local": MetadataTrackerLocal,
    "sql": MetadataTrackerSQL,
    "mongo": MetadataTrackerMongo
}


def get_metadata_tracker(config):
    """
    Based on config, get back the type of metadata tracker.
    Will check management.metadata_tracker_type or fallback on local

    Parameters
    ----------
    config a complete config (not flatted!)

    Returns
    -------
    Object that extends MetadataTrackerInterface

    """

    # Fallback mechanism: if management not defined, or if metadata_tracker in management not defined: use local
    metadata_tracker = config\
        .get("management", {})\
        .get("metadata_tracker_type", "local")

    return METADATA_TRACKER.get(metadata_tracker)(config)
