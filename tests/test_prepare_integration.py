from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mimic4_ihm.prepare import prepare


class PrepareIntegrationTests(unittest.TestCase):
    def test_small_csv_pipeline_writes_shared_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mimic"
            output = Path(temporary) / "output"
            (root / "hosp").mkdir(parents=True)
            (root / "icu").mkdir(parents=True)
            subjects = [100, 101, 102, 103]
            pd.DataFrame(
                {
                    "subject_id": subjects,
                    "gender": ["F", "M", "F", "M"],
                    "anchor_age": [50, 60, 70, 80],
                    "anchor_year": [2100] * 4,
                }
            ).to_csv(root / "hosp" / "patients.csv", index=False)
            pd.DataFrame(
                {
                    "subject_id": subjects,
                    "hadm_id": [200, 201, 202, 203],
                    "admittime": ["2100-01-01 00:00:00"] * 4,
                    "dischtime": ["2100-01-05 00:00:00"] * 4,
                    "deathtime": ["2100-01-05 12:00:00", "2100-01-04 12:00:00", None, None],
                    "hospital_expire_flag": [1, 1, 0, 0],
                    "insurance": ["Private", "Medicare", "Medicaid", "Other"],
                    "race": ["WHITE", "BLACK/AFRICAN AMERICAN", "ASIAN", "HISPANIC/LATINO"],
                }
            ).to_csv(root / "hosp" / "admissions.csv", index=False)
            pd.DataFrame(
                {
                    "subject_id": subjects,
                    "hadm_id": [200, 201, 202, 203],
                    "stay_id": [300, 301, 302, 303],
                    "first_careunit": ["MICU"] * 4,
                    "last_careunit": ["MICU"] * 4,
                    "intime": ["2100-01-01 06:00:00"] * 4,
                    "outtime": ["2100-01-04 06:00:00"] * 4,
                    "los": [3.0] * 4,
                }
            ).to_csv(root / "icu" / "icustays.csv", index=False)
            pd.DataFrame(
                {"itemid": [220045], "label": ["Heart Rate"], "unitname": ["bpm"]}
            ).to_csv(root / "icu" / "d_items.csv", index=False)
            pd.DataFrame({"itemid": [50931], "label": ["Glucose"]}).to_csv(
                root / "hosp" / "d_labitems.csv", index=False
            )
            chart_rows = []
            for stay_id in [300, 301, 302, 303]:
                chart_rows.extend(
                    [
                        {
                            "stay_id": stay_id,
                            "charttime": "2100-01-01 05:59:00",
                            "itemid": 220045,
                            "value": "70",
                            "valuenum": 70,
                        },
                        {
                            "stay_id": stay_id,
                            "charttime": "2100-01-01 07:00:00",
                            "itemid": 220045,
                            "value": "80",
                            "valuenum": 80,
                        },
                    ]
                )
            pd.DataFrame(chart_rows).to_csv(root / "icu" / "chartevents.csv", index=False)
            pd.DataFrame(
                {
                    "hadm_id": [200, 201, 202, 203],
                    "charttime": ["2100-01-01 08:00:00"] * 4,
                    "itemid": [50931] * 4,
                    "value": [100, 110, 120, 130],
                    "valuenum": [100, 110, 120, 130],
                }
            ).to_csv(root / "hosp" / "labevents.csv", index=False)

            prepare(
                argparse.Namespace(
                    mimic_root=root,
                    output_dir=output,
                    seed=49297,
                    max_stays=None,
                    subject_list=None,
                    chunksize=2,
                    overwrite=False,
                )
            )
            x = np.load(output / "lstm_X.npy", mmap_mode="r")
            lr = np.load(output / "lr_X_raw.npy", mmap_mode="r")
            metadata = pd.read_csv(output / "metadata.csv")
            self.assertEqual(x.shape, (4, 48, 76))
            self.assertEqual(lr.shape, (4, 714))
            self.assertEqual(metadata["label"].sum(), 1)
            self.assertEqual(metadata.loc[metadata["stay_id"].eq(300), "label"].item(), 0)
            self.assertEqual(set(metadata["split"]), {"train", "validation", "test"})
            self.assertTrue((output / "mapping_audit.csv").is_file())
            self.assertTrue((output / "demographic_mapping_audit.csv").is_file())
            self.assertTrue((output / "continuous_value_audit.csv").is_file())
            self.assertTrue((output / "coverage.csv").is_file())
            del x, lr


if __name__ == "__main__":
    unittest.main()
