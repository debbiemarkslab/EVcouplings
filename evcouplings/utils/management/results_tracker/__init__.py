from evcouplings.utils.management.results_tracker.ResultsTrackerLocal import ResultsTrackerLocal
from evcouplings.utils.management.results_tracker.ResultsTrackerNull import ResultsTrackerNull
from evcouplings.utils.management.results_tracker.ResultsTrackerMongo import ResultsTrackerMongo


# Dumper types. Fallback is "null"
RESULTS_TRTACKERS = {
    "local": ResultsTrackerLocal,
    "mongo": ResultsTrackerMongo,
    None: ResultsTrackerNull
}


def get_results_tracker(config):
    """
    Based on config, get back results_tracker (where to store zip file and/or run results and/or intermediate results)
    Will fallback to a ResultsTrackerNull, which won't do anything.

    Parameters
    ----------
    config A complete config (not flattened!)

    Returns
    -------
    Object extending ResultsTrackerInterface

    """

    # Fallback mechanism: if management not defined, or location in results_tracker not defined: use ResultsTrackerNull
    results_tracker = config \
        .get("management", {}) \
        .get("results_tracker_type")

    return RESULTS_TRTACKERS.get(results_tracker)(config)
