import io
import os
import sys
import numpy as np

import unittest
from unittest.mock import patch
import contextlib

## Fiora imports
import fiora.IO.mgfReader as mgfReader
from fiora.MS.spectral_scores import spectral_cosine

## Importing fiora predict (from executable)
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader 
spec = spec_from_loader("fiora-predict", SourceFileLoader("fiora-predict", os.getcwd() + "/scripts/fiora-predict"))
fiora_predict = module_from_spec(spec)
spec.loader.exec_module(fiora_predict)
sys.modules['fiora_predict'] = fiora_predict


class TestFioraPredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_path = "temp_spec.mgf"
    
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_path):
            os.remove(cls.temp_path)

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

    def test_model_cpu(self):
        f = io.StringIO()
        with patch("sys.argv", ["main", "-i", "examples/example_input.csv", "-o", self.temp_path]):
            with contextlib.redirect_stdout(f):
                fiora_predict.main()
            self.assertIn("Finished prediction.", f.getvalue())   
            self.assertTrue(os.path.exists(self.temp_path))

    def test_model_output_integrity(self):
        expected_output = "examples/expected_output.mgf"

        df_expected = mgfReader.read(expected_output, as_df=True)
        df_new = mgfReader.read(self.temp_path, as_df=True)
        
        columns = ["TITLE", "SMILES", "PRECURSORTYPE", "COLLISIONENERGY", "INSTRUMENTTYPE"]
        self.assertDictEqual(df_expected[columns].to_dict(), df_new[columns].to_dict())
        for i, data in df_expected.iterrows():
            peaks_expected = data["peaks"]
            peaks_new = df_new.at[i, "peaks"]
            cosine = spectral_cosine(peaks_expected, peaks_new, transform=np.sqrt)
            self.assertGreater(cosine, 0.999)

if __name__ == '__main__':
    # unittest.main()
    suite = unittest.TestSuite()
    
    suite.addTests([
        TestFioraPredict('test_dummy'),
        TestFioraPredict('test_help'),
        TestFioraPredict('test_missing_args'),
        TestFioraPredict('test_model_cpu'),
        TestFioraPredict('test_model_output_integrity'),
    ])
    
    runner = unittest.TextTestRunner()
    runner.run(suite)