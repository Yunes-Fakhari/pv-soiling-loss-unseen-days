import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from PIL import Image

from pbi_loss.config import DEFAULTS
from pbi_loss.data import parse_filename, normalize_metadata, subsample
from pbi_loss.metrics import regression_metrics
from pbi_loss.model import LossModel, sun_factor
from pbi_loss.pipeline import regional_index, extract_features
from pbi_loss.protocols import make_folds


class CoreTests(unittest.TestCase):
    def test_region_layout(self):
        shares = np.zeros((12, 6))
        shares[0, :2] = 0.7
        self.assertAlmostEqual(float(regional_index(shares.ravel())), 1/3)
        shares[0, 1], shares[0, 2] = 0, 0.7
        self.assertAlmostEqual(float(regional_index(shares.ravel())), 1/3)
        shares[0, :4] = 0.7
        self.assertAlmostEqual(float(regional_index(shares.ravel())), 2/3)
        self.assertEqual(float(regional_index(np.ones(72))), 1)
        self.assertEqual(float(regional_index(np.zeros(72))), 0)

    def test_recovers_known_formula(self):
        rng = np.random.default_rng(9)
        index, hours = rng.uniform(0, 1, 200), rng.uniform(6, 18, 200)
        labels = 0.5*index**0.7 + 0.2*index**0.7*sun_factor(hours) + 0.03
        model = LossModel.fit(index, hours, labels)
        np.testing.assert_allclose(model.coefficients, [0.5, 0.2, 0.03], atol=1e-12)
        np.testing.assert_allclose(model.predict(index, hours), labels, atol=1e-12)

    def test_rank_deficient_training_fails(self):
        with self.assertRaisesRegex(ValueError, 'rank deficient'):
            LossModel.fit(np.zeros(10), np.arange(6, 16), np.full(10, 0.1))

    def test_known_metrics(self):
        result = regression_metrics([0, 0.5, 1], [0.1, 0.5, 0.9])
        self.assertAlmostEqual(result['MSE'], 0.02/3)
        self.assertAlmostEqual(result['MAE'], 0.2/3)
        self.assertAlmostEqual(result['R2'], 0.96)
        self.assertIsNone(regression_metrics([0.5, 0.5], [0.4, 0.6])['R2'])

    def test_filename_timestamp_and_label(self):
        result = parse_filename('solar_Tue_Jun_13_9__46__49_2017_L_0.047_I_0.296.jpg')
        self.assertEqual(result['datetime'], pd.Timestamp('2017-06-13 09:46:49'))
        self.assertEqual(result['loss'], 0.047)

    def test_sampling_is_chronological(self):
        frame = pd.DataFrame({'relpath': ['b.jpg', 'a.jpg', 'c.jpg'],
                              'datetime': ['2017-06-13 09:00:10', '2017-06-13 09:00:00', '2017-06-13 09:00:30'],
                              'loss': [0.2, 0.1, 0.3]})
        self.assertEqual(subsample(frame).relpath.tolist(), ['a.jpg', 'c.jpg'])

    def test_held_out_days_and_determinism(self):
        timestamps = [f'2017-06-{day:02d} 09:00:00' for day in range(13, 28)]
        frame = normalize_metadata(pd.DataFrame({'relpath': [f'{i}.jpg' for i in range(15)],
                                                 'datetime': timestamps, 'loss': 0.1}))
        _, train, test = next(make_folds(frame, 'random_days', 42))
        self.assertFalse(set(frame.iloc[train].date) & set(frame.iloc[test].date))
        self.assertEqual(len(test), 3)
        np.testing.assert_array_equal(test, next(make_folds(frame, 'random_days', 42))[2])

    def test_invalid_reference_is_not_a_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'gray.png'
            Image.fromarray(np.full((192, 192, 3), 100, np.uint8)).save(path)
            with self.assertRaisesRegex(ValueError, 'blueness reference'):
                extract_features(path, DEFAULTS['processing'])

    def test_prediction_does_not_read_filename_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            rgb = np.zeros((192, 192, 3), dtype=np.uint8)
            rgb[:] = [20, 40, 120]
            one, two = Path(directory)/'L_0.png', Path(directory)/'L_1.png'
            Image.fromarray(rgb).save(one)
            Image.fromarray(rgb).save(two)
            a = extract_features(one, DEFAULTS['processing'])
            b = extract_features(two, DEFAULTS['processing'])
            self.assertIsInstance(a['corner_fallback'], bool)
            self.assertEqual(a['pbi'], b['pbi'])
            np.testing.assert_array_equal(a['shares'], b['shares'])


if __name__ == '__main__':
    unittest.main()
