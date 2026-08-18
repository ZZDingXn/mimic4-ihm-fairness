from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mimic4_ihm.constants import CHANNELS, build_header
from mimic4_ihm.prepare import (
    HOUR_NS,
    age_group,
    assign_splits,
    clean_chart_value,
    encode_stay,
    ethnicity_group,
    gender_group,
    insurance_group,
)


ROOT = Path(__file__).resolve().parents[1]


class HeaderTests(unittest.TestCase):
    def test_header_matches_original_benchmark(self) -> None:
        raw_path = os.environ.get("MIMIC3_DISCRETIZER_CONFIG")
        if not raw_path:
            self.skipTest(
                "Set MIMIC3_DISCRETIZER_CONFIG to run the optional upstream header-alignment test"
            )
        config_path = Path(raw_path)
        if not config_path.is_file():
            self.skipTest(f"Upstream benchmark config was not found: {config_path}")
        original = json.loads(config_path.read_text(encoding="utf-8"))
        expected: list[str] = []
        for channel in sorted(original["id_to_channel"]):
            values = original["possible_values"][channel]
            expected.extend([f"{channel}->{value}" for value in values] if values else [channel])
        expected.extend(f"mask->{channel}" for channel in sorted(original["id_to_channel"]))
        header, _, _ = build_header()
        self.assertEqual(header, expected)
        self.assertEqual(len(header), 76)


class CleaningTests(unittest.TestCase):
    def test_unit_conversions(self) -> None:
        self.assertAlmostEqual(clean_chart_value(223835, 50, 50)[0], 0.5)
        self.assertAlmostEqual(clean_chart_value(220277, 0.97, 0.97)[0], 97.0)
        self.assertAlmostEqual(clean_chart_value(223761, 98.6, 98.6)[0], 37.0)
        self.assertEqual(clean_chart_value(226707, 70, 70)[0], 178)
        self.assertAlmostEqual(clean_chart_value(226531, 220, 220)[0], 99.79024)

    def test_unknown_gcs_is_auditable(self) -> None:
        value, reason = clean_chart_value(220739, "unexpected", np.nan)
        self.assertIsNone(value)
        self.assertIn("unknown", reason)

    def test_gcs_numeric_valuenum_is_used_when_text_is_missing(self) -> None:
        self.assertEqual(clean_chart_value(220739, np.nan, 4.0), ("4 Spontaneously", None))
        self.assertEqual(clean_chart_value(223901, np.nan, 2.0), ("2 Abnorm extensn", None))
        self.assertEqual(clean_chart_value(223900, np.nan, 1.0), ("1 No Response", None))

    def test_demographic_groups(self) -> None:
        self.assertEqual(gender_group("F"), "Female")
        self.assertEqual(age_group(90), "90+")
        self.assertEqual(ethnicity_group("HISPANIC/LATINO - BLACK"), "Hispanic")
        self.assertEqual(ethnicity_group("ASIAN - CHINESE"), "Asian")
        self.assertEqual(insurance_group("Medicare"), "Medicare")
        self.assertEqual(insurance_group("No charge"), "Other")


class EncodingTests(unittest.TestCase):
    def test_boundaries_last_value_imputation_and_masks(self) -> None:
        intime = pd.Timestamp("2020-01-01 00:00:00")
        events = [
            (intime.value, "Heart Rate", 80.0, 220045, "chart"),
            (intime.value + HOUR_NS, "Heart Rate", 81.0, 220045, "chart"),
            (intime.value + HOUR_NS + 1_000_000_000, "Heart Rate", 82.0, 220045, "chart"),
            (intime.value + 48 * HOUR_NS, "Heart Rate", 83.0, 220045, "chart"),
        ]
        encoded, masks, lr_features = encode_stay(events, intime)
        header, _, _ = build_header()
        heart_rate = header.index("Heart Rate")
        heart_mask = CHANNELS.index("Heart Rate")
        self.assertEqual(encoded[0, heart_rate], 81.0)
        self.assertEqual(encoded[1, heart_rate], 82.0)
        self.assertEqual(encoded[47, heart_rate], 83.0)
        self.assertEqual(masks[0, heart_mask], 1.0)
        self.assertEqual(masks[1, heart_mask], 1.0)
        self.assertEqual(masks[47, heart_mask], 1.0)
        self.assertEqual(lr_features.shape, (714,))

    def test_normal_value_and_previous_imputation(self) -> None:
        intime = pd.Timestamp("2020-01-01")
        event_time = intime.value + 2 * HOUR_NS
        encoded, masks, _ = encode_stay(
            [(event_time, "Heart Rate", 90.0, 220045, "chart")], intime
        )
        header, _, _ = build_header()
        heart_rate = header.index("Heart Rate")
        self.assertEqual(encoded[0, heart_rate], 86.0)
        self.assertEqual(encoded[1, heart_rate], 90.0)
        self.assertEqual(encoded[2, heart_rate], 90.0)
        self.assertEqual(masks[:, CHANNELS.index("Heart Rate")].sum(), 1.0)


class SplitTests(unittest.TestCase):
    def test_patient_level_split_is_deterministic_and_disjoint(self) -> None:
        metadata = pd.DataFrame(
            {
                "subject_id": np.repeat(np.arange(100, 200), 2),
                "stay_id": np.arange(200),
            }
        )
        first = assign_splits(metadata, 49297)
        second = assign_splits(metadata, 49297)
        self.assertTrue(first.equals(second))
        joined = metadata.assign(split=first)
        self.assertTrue(joined.groupby("subject_id")["split"].nunique().eq(1).all())
        self.assertEqual(set(first.unique()), {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
