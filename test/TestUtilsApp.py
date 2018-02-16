import os
import unittest
from unittest import TestCase
from evcouplings.utils import SubmitterFactory, Command
from evcouplings.utils.app import app

class TestUtilsAppp(TestCase):

    def setUp(self):
        self.config_file = ""

    def test_substitute_config_ResourceError(self):
        """
        tests whether the ResourceError

        """


