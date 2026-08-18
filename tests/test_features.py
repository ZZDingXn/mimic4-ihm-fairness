from __future__ import annotations

import unittest

import numpy as np

from mimic4_ihm.features import extract_lr_features, lr_feature_names, numeric_event_value


class LogisticFeatureTests(unittest.TestCase):
    def test_feature_shape_and_names(self) -> None:
        features = extract_lr_features(
            [
                (0.0, "Heart Rate", 80.0),
                (24.0, "Heart Rate", 90.0),
                (48.0, "Heart Rate", 100.0),
            ]
        )
        self.assertEqual(features.shape, (714,))
        self.assertEqual(len(lr_feature_names()), 714)
        self.assertTrue(np.isnan(features).any())

    def test_categorical_values_become_numeric(self) -> None:
        self.assertEqual(numeric_event_value("Capillary refill rate", "1.0"), 1.0)
        self.assertEqual(
            numeric_event_value("Glascow coma scale eye opening", "4 Spontaneously"), 4.0
        )
        self.assertEqual(numeric_event_value("Glascow coma scale total", "15"), 15.0)


if __name__ == "__main__":
    unittest.main()

