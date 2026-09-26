"""The short fitting entry must not silently use the development template."""
from pathlib import Path
import tempfile
import unittest

from config import get_model_save_path
from reproduce import paper_sir_config


class PaperEntryTests(unittest.TestCase):
    def test_reported_cohort_solver_and_training_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = paper_sir_config(output)
            self.assertEqual(config.TRAIN_COUNTRIES,
                             ["DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "AUT", "CHE", "GBR"])
            self.assertFalse(set(config.TRAIN_COUNTRIES) & {"FIN", "NOR", "SWE", "CZE", "GRC"})
            self.assertEqual(config.SIR_RECOVERY_DAYS, 28)
            self.assertTrue(config.FREEZE_GAMMA)
            self.assertFalse(config.FAST_EULER_LOSS)
            self.assertEqual((config.SDE_SUBSTEP_DT, config.SDE_NUM_SUBSTEPS), (0.1, 10))
            self.assertEqual((config.SEED, config.NUM_EPOCHS, config.BATCH_SIZE), (42, 60, 512))
            self.assertEqual(config.NUM_ENSEMBLE_MODELS, 1)
            self.assertEqual(get_model_save_path(config).parent, output / "sir_fit")
            self.assertEqual(config.SNAPSHOT_DIR, output / "sir_fit" / "snapshots")
