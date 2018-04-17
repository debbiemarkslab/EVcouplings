import os
import unittest
from unittest import TestCase
from evcouplings.utils import check_required, MissingParameterError


class TestUtilsConfig(TestCase):
    def setUp(self):
        self.config = {
            "global": {
                "prefix": "test"
                      },
            "management": {
                 "database_uri": 'sqlite:///test.db',
                 "job_name": "test_case"
                           }
        }

    def test_check_required(self):
        """
        tests if function correctly checks parameter keys presence
        """
        check_required(self.config, self.config.keys())

    def test_check_required_error(self):
        """
        tests if function raises MissingParameterError if key is not present
        """
        self.assertRaises(MissingParameterError,
                          check_required, self.config,
                          "missing")



if __name__ == '__main__':
        unittest.main()
