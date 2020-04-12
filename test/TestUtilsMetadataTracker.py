import os
import unittest
from unittest import TestCase

from evcouplings.utils.management import get_metadata_tracker, EStatus
from evcouplings.utils.management.metadata_tracker import MetadataTrackerSQL


class TestUtilsMetadataTracker(TestCase):

    def setUp(self):
        self.config = {
            "global": {
                "prefix": "test"
            },
            "management": {
                "metadata_tracker_type": "sql",
                "metadata_tracker_uri": 'sqlite:///test.db',
                "job_name": "test_case"
            }
        }

    def test_update_job_status(self):
        """
        test whether job status is changed
        """
        mgmt = self.config.get("management", {})
        uri = mgmt.get("metadata_tracker_uri", None)
        job_name = mgmt.get("job_name", None)


        # Set up tracker object
        tracker = get_metadata_tracker(self.config)

        # Update status
        tracker.update_job_status(EStatus.RUN)

        job = MetadataTrackerSQL.get_job(job_name, uri)

        self.assertEqual(mgmt["job_name"], job["job_name"])
        self.assertEqual(EStatus.RUN, job["status"])

        # Update status
        tracker.update_job_status(EStatus.DONE)

        job_updated = MetadataTrackerSQL.get_job(job_name, uri)
        self.assertEqual(EStatus.DONE, job_updated["status"])

    def tearDown(self):
        os.remove("test.db")


if __name__ == '__main__':
    unittest.main()
