# Data Science Coursework: Applying the Data Science Lifecycle to Big Data

This is a shared repository holding **two independent coursework projects**, each completed individually by a different author on a different dataset. Both apply the **Data Science Lifecycle** (CRISP-DM style: problem identification, data engineering, EDA, modelling, recommendations) to large, real-world data. The projects do not depend on each other and can be run separately.

| Project | Domain | Dataset | Author / folder |
|---|---|---|---|
| **Predicting Repeat Purchases in Online Retail** | Retail business analytics | UCI Online Retail (541,909 rows) | Nethru Wickramasekara (`analysis_pipeline.py`) |
| **Movie Rating Prediction and Content Discovery** | Entertainment analytics | MovieLens 25M (25,000,095 ratings) | Hiruni Hapuarachchi (`movielens_analysis.py`) |

Each project was written up in a separate report (not included in this repo) following the same five-task structure:

1. Problem identification, stakeholder analysis and dataset justification
2. Data engineering and preprocessing
3. Exploratory data analysis and visualisation
4. Modelling and evaluation
5. Recommendation strategies and future decision-making

---

## Repository structure

```
Data-Science-Coursework/
├── analysis_pipeline.py        # Online Retail: cleaning, features, EDA, models (Nethru)
├── movielens_analysis.py       # MovieLens 25M: audit, EDA, models (Hiruni)
├── Online Retail.xlsx          # Online Retail source data
├── requirements.txt            # Python dependencies
├── notebooks/                  # Notebook versions of both scripts
│   ├── online_retail_analysis.ipynb
│   └── movielens_analysis.ipynb
├── tests/
│   └── test_smoke.py           # Smoke tests (see "Testing")
├── .gitattributes              # Git LFS rules for large data files
│
├── analysis_outputs/           # Online Retail outputs (Nethru)
│   ├── fig1_monthly_revenue.png ... fig6_permutation_importance.png
│   ├── model_metrics.csv, tuning_results.csv, permutation_importance.csv
│   ├── customer_snapshots.csv, snapshot_summary.csv, monthly_summary.csv
│   ├── country_summary.csv, top_products.csv
│   └── profile.json            # Dataset profile / data-quality audit
│
├── ml-25m/                     # MovieLens 25M raw data (GroupLens)
│   └── ratings.csv, movies.csv, tags.csv, links.csv,
│       genome-scores.csv, genome-tags.csv
└── movielens_outputs/          # MovieLens outputs (Hiruni)
    ├── fig1_rating_distribution.png ... fig6_model_rmse.png
    ├── model_results.csv       # Final model comparison table
    └── results.json            # Counts, correlations, tuning, metrics
```

| Project | Script | Input data | Outputs |
|---|---|---|---|
| Online Retail (Nethru) | `analysis_pipeline.py` | `Online Retail.xlsx` | `analysis_outputs/` |
| MovieLens 25M (Hiruni) | `movielens_analysis.py` | `ml-25m/` | `movielens_outputs/` |

---

## Project 1: Online Retail, predicting repeat purchases

*Author: Nethru Wickramasekara. Script: `analysis_pipeline.py`. Outputs: `analysis_outputs/`.*

**Goal:** At a monthly snapshot date, predict whether an identifiable customer will place at least one positive order in the next 30 days, to support campaign prioritisation.

**Data:** UCI Online Retail (Chen, 2015), 1 Dec 2010 to 9 Dec 2011. DOI: [10.24432/C5BW33](https://doi.org/10.24432/C5BW33). Licence: CC BY 4.0.

**Approach**

- **Cleaning:** removed 5,268 exact duplicates; excluded 135,080 rows with missing `CustomerID`, cancellations, and non-positive quantities or prices. Result: 392,692 lines, 18,532 invoices and 4,338 customers.
- **Features:** 15 customer-level features over a 180-day window (recency, orders, spend, unique products, order value, 30-day momentum, UK share and others), built for 8 monthly snapshots (Apr to Nov 2011), giving 21,613 customer snapshots.
- **Leakage control:** chronological split (train Apr to Aug, validate Sep, test Oct to Nov), with winsorisation limits learned on training data only.
- **Models:** regularised logistic regression and histogram gradient boosting, with class weighting and F1-optimised thresholds.

**Out-of-time test results (Oct to Nov 2011), as reported**

| Model | Balanced acc. | F1 | ROC-AUC | PR-AUC | Top-decile lift |
|---|---|---|---|---|---|
| Logistic regression | 0.6323 | 0.5662 | 0.7056 | 0.6039 | 2.25 |
| Histogram gradient boosting | 0.6449 | 0.5625 | 0.7110 | 0.6124 | 2.22 |

Order count over the previous 180 days was by far the most important feature. The write-up recommends tiered campaign prioritisation, a randomised controlled trial and ongoing monitoring.

---

## Project 2: MovieLens 25M, rating prediction and content discovery

*Author: Hiruni Hapuarachchi. Script: `movielens_analysis.py`. Outputs: `movielens_outputs/`.*

**Goal:** Estimate a user's rating (0.5 to 5.0) for a movie using only information available before a cut-off, and use it to inform personalised, popularity-aware recommendations.

**Data:** MovieLens 25M (GroupLens Research, 2019): 25,000,095 ratings, 62,423 movies, 162,541 users. Source: <https://grouplens.org/datasets/movielens/25m/>.

**Approach**

- **Audit:** checked duplicates, missing values, rating bounds and timestamps; no ratings were removed as outliers.
- **Features:** leave-one-out, smoothed (λ=10) user and movie means, log counts, release year, movie age and rating month.
- **Split:** chronological, with a cut-off of 25 June 2016 (20.0M train / 5.0M test). Modelling used deterministic samples of 2M train and 1M test rows (seed 42).
- **Models:** Ridge regression and histogram gradient boosting, compared against global-mean and smoothed movie-mean baselines.

**Future test results, as reported**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Global-mean baseline | 1.074827 | 0.838344 | -0.000957 |
| Smoothed movie-mean baseline | 1.007084 | 0.771134 | 0.121242 |
| Ridge regression | 1.031885 | 0.822164 | 0.077428 |
| Histogram gradient boosting | 1.006927 | 0.783870 | 0.121515 |

**Key finding:** gradient boosting is best on RMSE, but its edge over the movie-mean baseline is negligible (about 0.016%). With 84.7% of test rows from new users, cold start is the dominant issue. The write-up recommends keeping the baseline, building a segmented hybrid system and running an A/B test.

---

## Getting started

**Requirements:** Python 3.10+ (the MovieLens analysis was run on Python 3.13), with pandas, NumPy, scikit-learn, Matplotlib, Seaborn and openpyxl (for the `.xlsx` file).

```bash
git clone https://github.com/Nethru2002/Data-Science-Coursework.git
cd Data-Science-Coursework

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Run the Online Retail pipeline** (from the repo root)

```bash
python analysis_pipeline.py
```

Reads `Online Retail.xlsx` and writes figures and tables to `analysis_outputs/`.

**Run the MovieLens analysis** (from the repo root)

```bash
python movielens_analysis.py
```

Reads `ratings.csv`, `movies.csv` and `tags.csv` from `ml-25m/` (if that folder is missing, it extracts `ml-25m.zip` from the repo root) and writes figures, `model_results.csv` and `results.json` to `movielens_outputs/`.

> The full MovieLens ratings table needs several GB of RAM. The modelling step uses deterministic samples (2M training and 1M test rows), while the audit and EDA use all 25 million ratings.

---

## Notebooks

Step-by-step notebook versions of the two scripts are in `notebooks/`:

- `online_retail_analysis.ipynb` (Nethru)
- `movielens_analysis.ipynb` (Hiruni Hapuarachchi)

Open them in Jupyter or VS Code (run `pip install jupyter` first if needed). The MovieLens notebook needs several GB of RAM. They find the data from either the repo root or `notebooks/`. The Online Retail notebook is the original script split into cells, so it writes to `analysis_outputs/` just like `analysis_pipeline.py`. The MovieLens notebook is likewise the original script split into cells, and writes to `movielens_outputs/`.

## Testing

A small smoke test checks that the scripts and notebooks are valid Python, the dependencies import, the output files exist, and key numbers in the outputs match the reports (for example 21,613 customer snapshots and a MovieLens RMSE of about 1.0069). It does not retrain any models. Score checks allow a small tolerance for library-version differences.

```bash
python -m unittest discover -s tests -v
```

To also re-run the Online Retail pipeline end to end (about a minute; it runs in a temporary folder, so `analysis_outputs/` is not changed):

```bash
RUN_RETAIL=1 python -m unittest discover -s tests -v      # Windows PowerShell: $env:RUN_RETAIL=1
```

---

## Data

The datasets used by the scripts are included in this repository:

- **Online Retail:** `Online Retail.xlsx` in the repo root. Source: <https://doi.org/10.24432/C5BW33>, licensed CC BY 4.0.
- **MovieLens 25M:** the CSV files in `ml-25m/`. Source: <https://grouplens.org/datasets/movielens/25m/>.

Large files are tracked with **Git LFS** (see `.gitattributes`). Install it before cloning so the data downloads correctly:

```bash
git lfs install
git clone https://github.com/Nethru2002/Data-Science-Coursework.git
```

**MovieLens licence note:** the dataset belongs to GroupLens Research (University of Minnesota). Its terms require acknowledgement, prohibit redistribution without permission, and require separate permission for commercial use. It is included here for coursework reproducibility only. If the `ml-25m/` folder is not in the repo, download the dataset from the official page above, extract it into `ml-25m/` in the repo root (or place `ml-25m.zip` there), and cite Harper and Konstan (2015).

---

## Reproducibility

- Random seed is fixed at **42** in both projects.
- All splits are chronological to avoid look-ahead leakage.
- Data cleaning, features and logistic-regression results reproduce exactly across environments. Gradient-boosting scores can shift slightly between scikit-learn versions (for example PR-AUC around 0.610 to 0.612), so pin the versions in `requirements.txt` to the ones you used for the reports (`pip freeze`).
- Figures and statistics in the write-ups were generated from the code in this repository, with no third-party charts reused.
- MovieLens numerical outputs are stored in `movielens_outputs/results.json` and `movielens_outputs/model_results.csv`; Online Retail outputs are in `analysis_outputs/`.

## References

- Chen, D. (2015) *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
- GroupLens Research (2019) *MovieLens 25M Dataset*. University of Minnesota.
- Harper, F.M. and Konstan, J.A. (2015) 'The MovieLens datasets: history and context', *ACM TiiS*, 5(4). https://doi.org/10.1145/2827872
- Chapman, P. et al. (2000) *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS Inc.

## Authors

Each project is the individual work of its author:

- **Nethru Wickramasekara:** Online Retail project (`analysis_pipeline.py`, `analysis_outputs/`)
- **Hiruni Hapuarachchi:** MovieLens 25M project (`movielens_analysis.py`, `movielens_outputs/`)