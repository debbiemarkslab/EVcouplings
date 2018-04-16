import unittest
from unittest import TestCase
from evcouplings.utils import *
import tempfile


class TestUtilsHelpers(TestCase):

    def test_wrap(self):
        """
        Test whether string is correctly wrap
        """
        out = wrap("Test", width=2)
        self.assertEqual("Te\nst", out)

    def test_range_overlap_noOverlapPosNumber(self):
        """
        Test whether range overlaps are correctly calculated
        """
        overlap = range_overlap((1,2), (3,4))
        self.assertEqual(overlap, 0)

    def test_range_overlap_overlapPosNumber(self):
        """
        Test whether range overlaps are correctly calculated
        """
        overlap = range_overlap((1, 3), (2, 4))
        self.assertEqual(overlap, 1)

    def test_range_overlap_start_greater_end(self):
        """
        Test whether range overlaps are correctly calculated
        """
        self.assertRaises(InvalidParameterError, range_overlap, (-2, -4), (-3, -1))


class TestUtilsProgressbar(TestCase):

    def test_initiation(self):
        p = Progressbar(10, 10)

    def test_update(self):
        p = Progressbar(5, 5)
        for i in range(5):
            p.update(i)


class TestDefaultOrderdDict(TestCase):

    def test_defaultOrderedDict(self):
        """
        test if order is maintained
        """

        d = DefaultOrderedDict()
        d["one"] = 1
        d["a"] = 3
        d["two"] = 2
        self.assertEqual("DefaultOrderedDict([('one', 1), ('a', 3), ('two', 2)])", str(d))


class TestPersistentDict(TestCase):

    def setUp(self):
        self.tmp_db = tempfile.NamedTemporaryFile(delete=False)

    def test_add_element(self):
        """
        Tests whether adding an elements provokes to sync the dict with a file system (it should not)
        """
        d = PersistentDict(self.tmp_db.name)
        d["test"] = "insert"
        self.assertFalse(valid_file(self.tmp_db.name))

    def test_get_element(self):
        """
        Tests whether adding an elements provokes to sync the dict with a file system (it should not)
        """
        d = PersistentDict(self.tmp_db.name)
        d["test"] = "insert"
        self.assertEqual(d["test"], "insert")

    def test_sync(self):
        """
        Tests whether adding an elements provokes to sync the dict with a file system (it should not)
        """
        d = PersistentDict(self.tmp_db.name)
        d["test"] = "insert"
        d.sync()
        self.assertTrue(valid_file(self.tmp_db.name))

    def test_sync_empty(self):
        """
        Tests whether adding an elements provokes to sync the dict with a file system (it should not)
        """
        d = PersistentDict(self.tmp_db.name)
        d.sync()
        self.assertFalse(valid_file(self.tmp_db.name))

    def test_dump(self):
        """
        Tests whether adding an elements provokes to sync the dict with a file system (it should not)
        """
        d = PersistentDict(self.tmp_db.name)
        tmp2 = tempfile.NamedTemporaryFile(mode="w", delete=False)
        d["test"] = "insert"
        d.dump(tmp2)
        tmp2.close()
        self.assertTrue(valid_file(tmp2.name))
        os.unlink(tmp2.name)

    def test_dump_empty(self):
        """
        Tests whether adding an elements provokes to sync the dict with a file system (it should not)
        """
        d = PersistentDict(self.tmp_db.name)
        tmp2 = tempfile.NamedTemporaryFile(mode="w", delete=False)
        d.dump(tmp2)
        tmp2.close()
        self.assertFalse(valid_file(tmp2.name))
        os.unlink(tmp2.name)

    def test_load(self):
        """
        Tests whether adding an elements provokes to sync the dict with a file system (it should not)
        """
        d = PersistentDict(self.tmp_db.name)
        d["test"] = "insert"
        d.sync()
        d.close()

        d2 = PersistentDict(self.tmp_db.name)
        d2.load(self.tmp_db)

    def tearDown(self):
        os.unlink(self.tmp_db.name)


if __name__ == '__main__':
    unittest.main()
