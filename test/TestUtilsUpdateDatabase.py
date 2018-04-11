import os
import unittest
from evcouplings.utils.update_database import symlink_force, download_ftp_file
from unittest import TestCase
import tempfile


class TestUtilsUpdateDatabase(TestCase):

    def test_symlink_force_createSymlink(self):
        """
        tests whether a symlink is correctly created
        """

        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()
        symlink_suffix = "_symlink"
        symlink_force(tmp.name, tmp.name + symlink_suffix)

        self.assertTrue(os.path.exists(tmp.name + symlink_suffix))
        os.unlink(tmp.name)
        self.assertFalse(os.path.exists(tmp.name + symlink_suffix))
        os.remove(tmp.name + symlink_suffix)

    def test_symlink_force_Error(self):
        """
        Tests if symlink_force generates an OSError if target file does not exist
        """
        self.assertRaises(OSError,  symlink_force, "/asdasdad/asdjkdawd", "/sfjksf/symlink")


if __name__ == '__main__':
    unittest.main()
