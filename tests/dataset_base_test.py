import shutil
import tempfile
import unittest


class DatasetTestCase(unittest.TestCase):

    class_test_dir = None

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @classmethod
    def setUpClass(cls):
        cls.class_test_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.class_test_dir)