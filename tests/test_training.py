from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.main import dataset_info, dataset_samples  # noqa: E402
from app.training import train_and_evaluate  # noqa: E402


class DatasetTests(unittest.TestCase):
    def test_dataset_metadata_and_samples_use_the_real_data(self):
        details = dataset_info()
        sample = dataset_samples(limit=2)
        self.assertEqual(details["records"], 569)
        self.assertEqual(details["features_count"], 30)
        self.assertEqual(details["class_distribution"], {"Maligno": 212, "Benigno": 357})
        self.assertEqual(len(details["features"]), 30)
        self.assertIn("Erro padrão", details["features"][10]["description"])
        self.assertIn("raio", details["features"][0]["description"])
        self.assertEqual(len(sample["rows"]), 2)
        self.assertEqual(sample["rows"][0]["diagnostico"], "Maligno")


class TrainingTests(unittest.TestCase):
    def test_default_run_matches_report_and_writes_real_charts(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = train_and_evaluate(artifact_dir=temporary)

            self.assertEqual(result["counts"]["total"], 569)
            self.assertEqual(result["counts"]["train"], 455)
            self.assertEqual(result["counts"]["test"], 114)
            self.assertEqual(result["confusion_labels"], ["Maligno", "Benigno"])
            self.assertEqual(result["confusion_matrix"], [[39, 3], [4, 68]])
            self.assertAlmostEqual(result["metrics"]["accuracy"], 0.93859649, places=6)
            self.assertAlmostEqual(result["metrics"]["precision"], 0.90697674, places=6)
            self.assertAlmostEqual(result["metrics"]["recall"], 0.92857143, places=6)
            self.assertAlmostEqual(result["metrics"]["f1"], 0.91764706, places=6)
            self.assertAlmostEqual(result["metrics"]["cv_mean"], 0.9227, places=3)
            self.assertAlmostEqual(result["metrics"]["cv_std"], 0.0295, places=3)
            self.assertEqual(len(result["cv_scores"]), 5)
            self.assertEqual([r["class"] for r in result["classification_report"]], ["Maligno", "Benigno"])

            root = Path(temporary)
            charts = ["confusion-matrix.png", "cross-validation.png", "class-distribution.png", "metrics.png"]
            for filename in charts:
                content = (root / filename).read_bytes()
                self.assertTrue(content.startswith(b"\x89PNG\r\n\x1a\n"), filename)

            malignant = result["classification_report"][0]
            self.assertAlmostEqual(malignant["recall"], 39 / (39 + 3), places=6)
            self.assertEqual(result["class_summary"][0]["incorrect"], 3)

    def test_default_run_is_reproducible(self):
        first = train_and_evaluate()
        second = train_and_evaluate()
        self.assertEqual(first["confusion_matrix"], second["confusion_matrix"])
        self.assertEqual(first["cv_scores"], second["cv_scores"])
        self.assertEqual(first["metrics"], second["metrics"])


if __name__ == "__main__":
    unittest.main()
