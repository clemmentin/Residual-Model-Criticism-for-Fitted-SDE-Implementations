"""The release's modules should be safe to import as a single package."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ModuleImportTests(unittest.TestCase):
    def test_data_and_bi_calculations_do_not_import_training_or_plotting(self):
        root = Path(__file__).resolve().parents[1]
        program = '''
import sys
sys.path.insert(0, sys.argv[1])
import sir_data
import brownian_inversion
for name in ("train_sde", "sir_training", "sir_experiments", "visualization", "matplotlib.pyplot"):
    assert name not in sys.modules, name
'''
        result = subprocess.run([sys.executable, "-c", program, str(root)],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_calibration_reporting_does_not_import_the_training_runner(self):
        root = Path(__file__).resolve().parents[1]
        program = '''
import sys
sys.path.insert(0, sys.argv[1])
import experiments.summarize_full_refit_calibration
assert "train_sde" not in sys.modules
assert "experiments.bi_synthetic_full_refit_internal_pvalue_calibration" not in sys.modules
'''
        result = subprocess.run([sys.executable, "-c", program, str(root)],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_all_diagnostics_import_without_changing_cwd_or_search_path(self):
        root = Path(__file__).resolve().parents[1]
        program = '''
import importlib, json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
cwd, search = os.getcwd(), list(sys.path)
for path in sorted((root / "experiments").glob("*.py")):
    if path.stem != "__init__":
        importlib.import_module("experiments." + path.stem)
        assert os.getcwd() == cwd, path.name
        assert sys.path == search, path.name
        assert path.stem not in sys.modules, path.name
assert "train_sde" not in sys.modules
from config import Config
assert Config().CACHE_DIR == root / "cache"
print("All diagnostic imports are canonical and leave cwd/sys.path unchanged.")
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-c", program, str(root)], cwd=directory,
                                    env={**os.environ, "PYTHONUTF8": "1"},
                                    capture_output=True, text=True, encoding="utf-8", timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
