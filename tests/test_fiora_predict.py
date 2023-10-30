import io
import os
import sys

import unittest
from unittest.mock import patch
import contextlib

## Importing fiora predict (from executable)
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader 
spec = spec_from_loader("fiora-predict", SourceFileLoader("fiora-predict", os.getcwd() + "/scripts/fiora-predict"))
fiora_predict = module_from_spec(spec)
spec.loader.exec_module(fiora_predict)
sys.modules['fiora_predict'] = fiora_predict


class TestFioraPredict(unittest.TestCase):

    def test_missing_args(self):
        f = io.StringIO()
        with patch("sys.argv", ["main"]):
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(f):
                fiora_predict.main()
            self.assertEqual(cm.exception.code, 2)
            self.assertTrue(f.getvalue().startswith("usage:"))   
            
            
    def test_help(self):
        f = io.StringIO()
        with patch("sys.argv", ["main", "-h"]):
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stdout(f):
                fiora_predict.main()
            self.assertEqual(cm.exception.code, 0)
            self.assertTrue(f.getvalue().startswith("usage:"))
            self.assertTrue("-h, --help" in  f.getvalue())
            self.assertTrue("show this help message and exit" in  f.getvalue())


    def test_dummy(self):
        self.assertEqual('fiora'.upper(), 'FIORA')

    def test_example_prediction_not_yet_implemented(self):
        self.assertTrue(1 == int("1"))
        self.assertFalse(2 == 1)


if __name__ == '__main__':
    unittest.main()