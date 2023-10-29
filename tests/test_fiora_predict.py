import os
import sys
import fiora.IO.mgfReader as mgfReader
from unittest.mock import patch
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader 

## Importing fiora predict
spec = spec_from_loader("fiora-predict", SourceFileLoader("fiora-predict", os.getcwd() + "/scripts/fiora-predict"))
fiora_predict = module_from_spec(spec)
spec.loader.exec_module(fiora_predict)
sys.modules['fiora_predict'] = fiora_predict
from fiora_predict import main


def test_help():
    with patch("sys.argv", ["main", "-h"]):
        main()
        captured = capsys.readouterr()
        
test_help()