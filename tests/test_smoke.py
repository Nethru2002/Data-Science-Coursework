import ast
import csv
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RETAIL_OUT = ROOT / "analysis_outputs"
ML_OUT = ROOT / "movielens_outputs"
NOTEBOOKS = ("online_retail_analysis.ipynb", "movielens_analysis.ipynb")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class TestCodeIsValid(unittest.TestCase):
    def test_scripts_parse(self):
        for name in ("analysis_pipeline.py", "movielens_analysis.py"):
            with self.subTest(script=name):
                path = ROOT / name
                self.assertTrue(path.exists(), f"{name} is missing")
                ast.parse(path.read_text(encoding="utf-8"), filename=name)

    def test_notebooks_are_valid(self):
        for name in NOTEBOOKS:
            with self.subTest(notebook=name):
                path = ROOT / "notebooks" / name
                self.assertTrue(path.exists(), f"notebooks/{name} is missing")
                nb = load_json(path)
                code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
                self.assertTrue(code.strip(), "notebook has no code cells")
                ast.parse(code, filename=name)
                self.assertNotIn("__file__", code)

    def test_dependencies_import(self):
        for mod in ("pandas", "numpy", "sklearn", "matplotlib", "seaborn", "openpyxl"):
            with self.subTest(module=mod):
                importlib.import_module(mod)


class TestInputsAndOutputsExist(unittest.TestCase):
    def test_online_retail_input(self):
        self.assertTrue((ROOT / "Online Retail.xlsx").exists(),
                        "Online Retail.xlsx is missing from the repo root")

    def test_online_retail_outputs(self):
        expected = [
            "country_summary.csv", "customer_snapshots.csv", "model_metrics.csv",
            "monthly_summary.csv", "permutation_importance.csv", "profile.json",
            "snapshot_summary.csv", "top_products.csv", "tuning_results.csv",
            "fig1_monthly_revenue.png", "fig2_country_revenue.png",
            "fig3_line_value_distribution.png", "fig4_spearman_heatmap.png",
            "fig5_target_imbalance.png", "fig6_permutation_importance.png",
        ]
        for name in expected:
            with self.subTest(file=name):
                self.assertTrue((RETAIL_OUT / name).exists())

    def test_movielens_outputs(self):
        expected = [
            "results.json", "model_results.csv", "fig1_rating_distribution.png",
            "fig2_ratings_by_year.png", "fig3_movie_long_tail.png",
            "fig4_genre_volume_mean.png", "fig5_user_activity.png", "fig6_model_rmse.png",
        ]
        for name in expected:
            with self.subTest(file=name):
                self.assertTrue((ML_OUT / name).exists())


class TestResultsMatchReports(unittest.TestCase):
    def test_online_retail_profile(self):
        p = load_json(RETAIL_OUT / "profile.json")
        self.assertEqual(p["rows"], 541909)
        self.assertEqual(p["exact_duplicates"], 5268)
        self.assertEqual(p["clean_positive_identified_rows"], 392692)
        self.assertEqual(p["clean_customers"], 4338)
        self.assertEqual(p["model_rows"], 21613)

    def test_online_retail_boosting_test_metrics(self):
        rows = read_csv(RETAIL_OUT / "model_metrics.csv")
        row = next(r for r in rows
                   if r["model"] == "Histogram gradient boosting" and r["split"] == "test")
        self.assertAlmostEqual(float(row["pr_auc"]), 0.6124, delta=5e-3)
        self.assertAlmostEqual(float(row["roc_auc"]), 0.7110, delta=5e-3)
        self.assertAlmostEqual(float(row["top_decile_lift"]), 2.2213, delta=5e-2)

    def test_movielens_results(self):
        r = load_json(ML_OUT / "results.json")
        self.assertEqual(r["ratings_rows"], 25000095)
        self.assertEqual(r["users"], 162541)
        hgb = next(m for m in r["model_results"] if m["model"] == "Histogram gradient boosting")
        self.assertAlmostEqual(hgb["RMSE"], 1.006927, delta=2e-3)
        self.assertLess(hgb["RMSE"], 1.074827 * 0.95)


@unittest.skipUnless(os.environ.get("RUN_RETAIL") == "1", "set RUN_RETAIL=1 to re-run the pipeline")
class TestRetailPipelineRuns(unittest.TestCase):
    def test_pipeline_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shutil.copy(ROOT / "analysis_pipeline.py", tmp)
            shutil.copy(ROOT / "Online Retail.xlsx", tmp)
            proc = subprocess.run([sys.executable, "analysis_pipeline.py"],
                                  cwd=tmp, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            p = load_json(tmp / "analysis_outputs" / "profile.json")
            self.assertEqual(p["rows"], 541909)
            self.assertEqual(p["clean_positive_identified_rows"], 392692)
            self.assertEqual(p["model_rows"], 21613)


if __name__ == "__main__":
    unittest.main()