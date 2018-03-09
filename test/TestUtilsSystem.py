import unittest
from unittest import TestCase
from evcouplings.utils import *


class TestUtilsApp(TestCase):

    def setUp(self):
        self.config_file = ""

    def test_run_ExternalToolError(self):
        """
        tests whether the ExternalToolError is properly raised

        assuming we are running on a unix based operation system
        """
        self.assertRaises(ExternalToolError, run, "testing")

    def test_run_returncode_ExternalToolError(self):
        """
        tests whether the ExternalToolError is properly raised when specifying returncode and
        command failed

        assuming we are running on a unix based operation system
        """
        with self.assertRaises(Exception) as context:
            run("ls /asdada", check_returncode=True, shell=True)
        self.assertTrue("Call failed:" in str(context.exception))

    def test_run(self):
        """
        tests whether the run works correctly
        """
        self.assertEqual((0, 'test\n', ''), run("echo test", shell=True))

    def test_valid_file_FileWithContent(self):
        """
        test if a file with content returns True
        """

        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(b"Testasdklasdaklndaklndakldnsalkdnasdnalsdkald")
        tmp.close()
        self.assertTrue(valid_file(tmp.name))
        os.unlink(tmp.name)

    def test_valid_file_EmptyFile(self):
        """
        test if a file with content returns False because file is empty
        """
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()
        self.assertFalse(valid_file(tmp.name))
        os.unlink(tmp.name)

    def test_valid_file_Error(self):
        """
        test if a file with content returns False because could not be found
        """
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()
        self.assertFalse(valid_file(tmp.name+"asdja"))
        os.unlink(tmp.name)

    def test_verify_resources_EmptyFileError(self):
        """
        Test if verify_resources returns ResourceError if files are empty
        """
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()

        f = [tmp.name]
        with self.assertRaises(Exception) as context:
            verify_resources("Verify_resources_empty", *f)
        self.assertTrue("Verify_resources_empty" in str(context.exception))
        os.unlink(tmp.name)

    def test_verify_resources_NonExistentFileError(self):
        """
        Test if verify_resources returns ResourceError if files are empty
        """
        f = ["asdfadasdjghfaqzfg"]

        with self.assertRaises(Exception) as context:
            verify_resources("Verify_resources_nonExistent", *f)
        self.assertTrue("Verify_resources_nonExistent" in str(context.exception))

    # create_prefix_folder: calls makedir, which internally calls os.makedir, so no need for a unittest

    def test_insert_dir_withRootDir(self):
        """
        test whether insert_dir generates a simple path directory /my/path/*dirs/prefix
        """
        t = ["test"]
        out = insert_dir("/my/path/prefix", *t)
        self.assertEqual(out, "/my/path/prefix/test/prefix")

    def test_insert_dir_noRootDir(self):
        """
        test whether insert_dir generates a simple path directory /my/path/*dirs/prefix
        """
        t = ["test"]
        out = insert_dir("/my/path/prefix", *t, rootname_subdir=False)
        self.assertEqual(out, "/my/path/test/prefix")

    # temp creates a tmp file, tempdir creates a tmp dir via the tempfile module, so no need for tests
    # write_file is just opening a file and writing content to it, no need for testing

    def test_get_Error(self):
        """
        Tests whether get generates a ResourceError we no request can be made
        """
        self.assertRaises(ResourceError, get, "")

    def test_get_InvalidStatusCodeError(self):
        """
        Tests whether get generates a ResourceError we no request can be made
        """
        self.assertRaises(ResourceError, get, "http://www.google.com/asdfsf")

    def test_get_NoOutputFileError(self):
        """
        Tests whether get generates a ResourceError we no request can be made
        """
        self.assertRaises(ResourceError, get, "http://www.google.com/", output_path="/afasjkd/asdaö")

    # get_urllib is using standard functions without additional logic so no need to test...


if __name__ == '__main__':
    unittest.main()
