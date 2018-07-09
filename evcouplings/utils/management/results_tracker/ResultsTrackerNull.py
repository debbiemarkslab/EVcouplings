"""
Zero/Null extension of results results_tracker:
will do absolutely nothing.

Used as fallback (and for intellectual consistency) if no results_tracker is specified.

Authors:
  Christian Dallago
"""

from evcouplings.utils.management.results_tracker.ResultsTrackerInterface import ResultsTrackerInterface


class ResultsTrackerNull(ResultsTrackerInterface):

    def __init__(self, config):
        super(ResultsTrackerNull, self).__init__(config)

    def write_file(self, _):
        return None

    def move_out_config_files(self, _):
        return None

    def clear(self):
        return None
