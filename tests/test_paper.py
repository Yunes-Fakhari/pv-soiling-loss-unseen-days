from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from pbi_loss.config import load_config
from pbi_loss.data import load_features
from pbi_loss.experiments import run_protocol
from pbi_loss.model import LossModel
from pbi_loss.protocols import make_folds

ROOT = Path(__file__).resolve().parents[1]


class PaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT/'configs/paper.json')
        cls.frame = load_features(ROOT/'data/reference/paper_features.csv.gz', cls.config['processing'])

    def test_manuscript_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            records = run_protocol(self.frame, self.config, directory, 'lodo')
        self.assertEqual(records[0]['n'], 11632)
        self.assertEqual(records[1]['n'], 9902)
        self.assertAlmostEqual(records[0]['MSE'], 0.01570957423561426, places=12)
        self.assertAlmostEqual(records[0]['R2'], 0.796271528277, places=10)

    def test_test_labels_do_not_change_fit(self):
        _, train, test = next(make_folds(self.frame, 'random_days', 42))
        changed = self.frame.copy()
        changed.loc[test, 'loss'] = 1-changed.loc[test, 'loss']
        def fit(frame):
            selected = frame.iloc[train]
            return LossModel.fit(selected.pbi, selected.hour_decimal, selected.loss)
        np.testing.assert_array_equal(fit(self.frame).coefficients, fit(changed).coefficients)


if __name__ == '__main__':
    unittest.main()
