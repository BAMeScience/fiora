import io
import os
import sys

import unittest
from unittest.mock import patch
import contextlib

from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader

spec = spec_from_loader(
    "fiora-model-info",
    SourceFileLoader("fiora-model-info", os.getcwd() + "/scripts/fiora-model-info"),
)
fiora_model_info = module_from_spec(spec)
spec.loader.exec_module(fiora_model_info)
sys.modules["fiora_model_info"] = fiora_model_info


class TestFioraModelInfo(unittest.TestCase):
    def test_missing_args(self):
        f = io.StringIO()
        with patch("sys.argv", ["main"]):
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(f):
                fiora_model_info.main()
            self.assertEqual(cm.exception.code, 2)
            self.assertTrue(f.getvalue().startswith("usage:"))

    def test_help(self):
        f = io.StringIO()
        with patch("sys.argv", ["main", "-h"]):
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stdout(f):
                fiora_model_info.main()
            self.assertEqual(cm.exception.code, 0)
            self.assertTrue(f.getvalue().startswith("usage:"))
            self.assertTrue("--model MODEL" in f.getvalue())
            self.assertTrue("--as-json" in f.getvalue())


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTests(
        [
            TestFioraModelInfo("test_missing_args"),
            TestFioraModelInfo("test_help"),
        ]
    )
    runner = unittest.TextTestRunner()
    runner.run(suite)
