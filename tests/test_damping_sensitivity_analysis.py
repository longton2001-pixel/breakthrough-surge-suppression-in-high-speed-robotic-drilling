import os
import sys
import unittest

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

import damping_sensitivity_analysis as dsa


class DampingSensitivityAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.df = dsa.load_validation_data()

    def test_uses_dedicated_damping_sensitivity_dataset(self):
        self.assertEqual(
            os.path.normpath(str(dsa.DEFAULT_DATA_PATH)),
            os.path.normpath(os.path.join(BASE_DIR, "data", "damping_sensitivity_data.csv")),
        )
        self.assertTrue(os.path.exists(dsa.DEFAULT_DATA_PATH))

    def test_baseline_eta_reproduces_existing_theoretical_velocity(self):
        predicted = dsa.predict_surge_velocity(
            self.df["theory^2"].to_numpy(),
            self.df["v_feed"].to_numpy(),
            dsa.ETA0,
        )

        np.testing.assert_allclose(
            predicted,
            self.df["v_surge_theory"].to_numpy(),
            rtol=0.0,
            atol=1e-6,
        )

    def test_constant_eta_sensitivity_preserves_rank_order(self):
        rows = dsa.run_constant_eta_sensitivity(self.df)

        self.assertEqual(len(rows), 5)
        self.assertEqual(
            [row["eta_setting"] for row in rows],
            ["0.8eta0", "0.9eta0", "1.0eta0", "1.1eta0", "1.2eta0"],
        )
        for row in rows:
            self.assertEqual(row["case"], "constant_eta")
            self.assertEqual(row["trials"], 1)
            self.assertAlmostEqual(row["spearman_mean"], 1.0, places=12)
            self.assertAlmostEqual(row["spearman_std"], 0.0, places=12)
            self.assertGreater(row["mean_predicted_v_surge_mm_s"], 0.0)

    def test_random_eta_sensitivity_is_deterministic_and_valid(self):
        first = dsa.run_random_eta_sensitivity(self.df)
        second = dsa.run_random_eta_sensitivity(self.df)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(
            [row["eta_setting"] for row in first],
            ["+/-10%", "+/-20%", "+/-30%"],
        )
        for row in first:
            self.assertEqual(row["case"], "posture_wise_random_eta")
            self.assertEqual(row["trials"], 1000)
            self.assertGreaterEqual(row["spearman_min"], -1.0)
            self.assertLessEqual(row["spearman_max"], 1.0)
            self.assertGreater(row["spearman_mean"], 0.7)

    def test_output_table_contains_required_columns(self):
        table = dsa.build_sensitivity_table(self.df)

        expected_columns = [
            "case",
            "eta_setting",
            "trials",
            "spearman_mean",
            "spearman_std",
            "spearman_min",
            "spearman_p05",
            "spearman_p95",
            "spearman_max",
            "mean_predicted_v_surge_mm_s",
        ]
        for column in expected_columns:
            self.assertIn(column, table.columns)
        self.assertEqual(len(table), 8)
        self.assertTrue(pd.api.types.is_numeric_dtype(table["spearman_mean"]))


if __name__ == "__main__":
    unittest.main()
