from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, brier_score_loss
)
from sklearn.inspection import permutation_importance


warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Online Retail.xlsx"
OUT = ROOT / "analysis_outputs"
OUT.mkdir(exist_ok=True)


def metric_row(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "tn": int(confusion_matrix(y, pred)[0, 0]),
        "fp": int(confusion_matrix(y, pred)[0, 1]),
        "fn": int(confusion_matrix(y, pred)[1, 0]),
        "tp": int(confusion_matrix(y, pred)[1, 1]),
    }


def top_decile_lift(y, p):
    n = max(1, int(np.ceil(len(y) * 0.10)))
    idx = np.argsort(-p)[:n]
    return float(np.mean(np.asarray(y)[idx]) / np.mean(y)), float(np.mean(np.asarray(y)[idx]))


print("Loading workbook...")
raw = pd.read_excel(DATA, engine="openpyxl")
raw["InvoiceNo"] = raw["InvoiceNo"].astype(str)
raw["StockCode"] = raw["StockCode"].astype(str)
raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
raw["CustomerID"] = pd.to_numeric(raw["CustomerID"], errors="coerce")
raw["LineAmount"] = raw["Quantity"] * raw["UnitPrice"]

profile = {
    "rows": int(len(raw)),
    "columns": int(raw.shape[1] - 1),
    "memory_mb": float(raw.drop(columns="LineAmount").memory_usage(deep=True).sum() / 1024**2),
    "date_min": str(raw.InvoiceDate.min()),
    "date_max": str(raw.InvoiceDate.max()),
    "unique_invoices": int(raw.InvoiceNo.nunique()),
    "unique_products": int(raw.StockCode.nunique()),
    "unique_customers": int(raw.CustomerID.nunique()),
    "unique_countries": int(raw.Country.nunique()),
    "missing": {k: int(v) for k, v in raw.isna().sum().to_dict().items()},
    "missing_pct": {k: float(v) for k, v in (raw.isna().mean() * 100).to_dict().items()},
    "exact_duplicates": int(raw.drop(columns="LineAmount").duplicated().sum()),
    "cancel_line_count": int(raw.InvoiceNo.str.startswith("C").sum()),
    "nonpositive_quantity": int((raw.Quantity <= 0).sum()),
    "nonpositive_price": int((raw.UnitPrice <= 0).sum()),
    "quantity_min": int(raw.Quantity.min()),
    "quantity_max": int(raw.Quantity.max()),
    "unitprice_min": float(raw.UnitPrice.min()),
    "unitprice_max": float(raw.UnitPrice.max()),
}

dedup = raw.drop(columns="LineAmount").drop_duplicates().copy()
clean = dedup[
    dedup.CustomerID.notna()
    & ~dedup.InvoiceNo.str.startswith("C")
    & (dedup.Quantity > 0)
    & (dedup.UnitPrice > 0)
].copy()
clean["CustomerID"] = clean.CustomerID.astype(int)
clean["LineAmount"] = clean.Quantity * clean.UnitPrice
clean["IsUK"] = (clean.Country == "United Kingdom").astype(int)

profile.update({
    "rows_after_dedup": int(len(dedup)),
    "clean_positive_identified_rows": int(len(clean)),
    "clean_customers": int(clean.CustomerID.nunique()),
    "clean_invoices": int(clean.InvoiceNo.nunique()),
    "clean_revenue": float(clean.LineAmount.sum()),
    "clean_median_line_amount": float(clean.LineAmount.median()),
    "clean_mean_line_amount": float(clean.LineAmount.mean()),
    "clean_line_amount_p99": float(clean.LineAmount.quantile(.99)),
    "uk_row_share": float(clean.IsUK.mean()),
})

monthly = (clean.assign(Month=clean.InvoiceDate.dt.to_period("M").astype(str))
           .groupby("Month").agg(revenue=("LineAmount", "sum"),
                                  invoices=("InvoiceNo", "nunique"),
                                  customers=("CustomerID", "nunique"))
           .reset_index())
monthly.to_csv(OUT / "monthly_summary.csv", index=False)
country = (clean.groupby("Country").agg(revenue=("LineAmount", "sum"),
                                         invoices=("InvoiceNo", "nunique"),
                                         customers=("CustomerID", "nunique"))
           .sort_values("revenue", ascending=False).reset_index())
country.to_csv(OUT / "country_summary.csv", index=False)
products = (clean.groupby(["StockCode", "Description"], dropna=False)
            .agg(quantity=("Quantity", "sum"), revenue=("LineAmount", "sum"),
                 invoices=("InvoiceNo", "nunique"))
            .sort_values("revenue", ascending=False).reset_index())
products.head(20).to_csv(OUT / "top_products.csv", index=False)

snapshots = pd.date_range("2011-04-01", "2011-11-01", freq="MS")
all_first = clean.groupby("CustomerID").InvoiceDate.min()
rows = []
for snap in snapshots:
    history = clean[(clean.InvoiceDate < snap) & (clean.InvoiceDate >= snap - pd.Timedelta(days=180))]
    eligible = all_first[all_first < snap].index
    hist = history[history.CustomerID.isin(eligible)].copy()
    grp = hist.groupby("CustomerID")
    f = grp.agg(
        last_purchase=("InvoiceDate", "max"),
        first_window_purchase=("InvoiceDate", "min"),
        line_items=("StockCode", "size"),
        orders=("InvoiceNo", "nunique"),
        unique_products=("StockCode", "nunique"),
        monetary=("LineAmount", "sum"),
        units=("Quantity", "sum"),
        mean_unit_price=("UnitPrice", "mean"),
        country_count=("Country", "nunique"),
        uk_share=("IsUK", "mean"),
    )
    f["recency_days"] = (snap - f.last_purchase).dt.total_seconds() / 86400
    f["active_span_days"] = ((f.last_purchase - f.first_window_purchase).dt.total_seconds() / 86400).clip(lower=0)
    f["avg_order_value"] = f.monetary / f.orders
    f["avg_units_per_order"] = f.units / f.orders
    f["purchase_rate_30d"] = f.orders / 6.0
    recent30 = history[history.InvoiceDate >= snap - pd.Timedelta(days=30)].groupby("CustomerID").agg(
        orders_30d=("InvoiceNo", "nunique"), spend_30d=("LineAmount", "sum"))
    f = f.join(recent30).fillna({"orders_30d": 0, "spend_30d": 0})
    future_ids = set(clean[(clean.InvoiceDate >= snap) &
                           (clean.InvoiceDate < snap + pd.Timedelta(days=30))].CustomerID.unique())
    f["target"] = f.index.to_series().map(lambda x: int(x in future_ids))
    f["snapshot"] = snap
    f["customer_id"] = f.index
    rows.append(f.reset_index(drop=True))

model_df = pd.concat(rows, ignore_index=True)
drop_cols = ["last_purchase", "first_window_purchase", "customer_id", "snapshot", "target"]
feature_cols = [c for c in model_df.columns if c not in drop_cols]

train = model_df[model_df.snapshot <= "2011-08-01"].copy()
valid = model_df[model_df.snapshot == "2011-09-01"].copy()
test = model_df[model_df.snapshot >= "2011-10-01"].copy()
lo = train[feature_cols].quantile(.01)
hi = train[feature_cols].quantile(.99)
for frame in (train, valid, test):
    frame.loc[:, feature_cols] = frame[feature_cols].clip(lo, hi, axis=1)

snapshot_summary = model_df.groupby("snapshot").target.agg(["count", "sum", "mean"]).reset_index()
snapshot_summary.to_csv(OUT / "snapshot_summary.csv", index=False)
profile["model_rows"] = int(len(model_df))
profile["model_positive_rate"] = float(model_df.target.mean())
profile["train_rows"] = int(len(train)); profile["valid_rows"] = int(len(valid)); profile["test_rows"] = int(len(test))
profile["train_positive_rate"] = float(train.target.mean()); profile["valid_positive_rate"] = float(valid.target.mean()); profile["test_positive_rate"] = float(test.target.mean())

Xtr, ytr = train[feature_cols], train.target
Xv, yv = valid[feature_cols], valid.target
Xte, yte = test[feature_cols], test.target

logit = Pipeline([
    ("prep", ColumnTransformer([("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())]), feature_cols)])),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=42))
])
logit.fit(Xtr, ytr)

grid_results = []
best = None
for lr in [0.03, 0.06, 0.10]:
    for leaves in [15, 31]:
        for leaf_n in [20, 50]:
            m = HistGradientBoostingClassifier(
                learning_rate=lr, max_iter=250, max_leaf_nodes=leaves,
                min_samples_leaf=leaf_n, l2_regularization=1.0,
                class_weight="balanced", early_stopping=False, random_state=42)
            m.fit(Xtr, ytr)
            pv = m.predict_proba(Xv)[:, 1]
            score = average_precision_score(yv, pv)
            grid_results.append({"learning_rate": lr, "max_leaf_nodes": leaves,
                                 "min_samples_leaf": leaf_n, "validation_pr_auc": score})
            if best is None or score > best[0]:
                best = (score, m, (lr, leaves, leaf_n))
boost = best[1]
pd.DataFrame(grid_results).sort_values("validation_pr_auc", ascending=False).to_csv(OUT / "tuning_results.csv", index=False)
profile["best_boost_params"] = {"learning_rate": best[2][0], "max_leaf_nodes": best[2][1], "min_samples_leaf": best[2][2]}

models = {"Logistic regression": logit, "Histogram gradient boosting": boost}
results = []
predictions = {}
for name, model in models.items():
    pv = model.predict_proba(Xv)[:, 1]
    pt = model.predict_proba(Xte)[:, 1]
    predictions[name] = pt
    precision, recall, thresholds = precision_recall_curve(yv, pv)
    f1s = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    ix = int(np.nanargmax(f1s[:-1]))
    threshold = float(thresholds[ix])
    for split_name, y, p in [("validation", yv, pv), ("test", yte, pt)]:
        r = metric_row(y, p, threshold)
        lift, capture_rate = top_decile_lift(y, p)
        r.update({"model": name, "split": split_name, "top_decile_lift": lift,
                  "top_decile_response_rate": capture_rate})
        results.append(r)
metrics = pd.DataFrame(results)
metrics.to_csv(OUT / "model_metrics.csv", index=False)

perm = permutation_importance(boost, Xte, yte, scoring="average_precision", n_repeats=10, random_state=42)
importance = pd.DataFrame({"feature": feature_cols, "importance_mean": perm.importances_mean,
                           "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
importance.to_csv(OUT / "permutation_importance.csv", index=False)

cust = clean.groupby("CustomerID").agg(revenue=("LineAmount", "sum"), orders=("InvoiceNo", "nunique"))
cust = cust.sort_values("revenue", ascending=False)
profile["top_1pct_customer_revenue_share"] = float(cust.head(max(1, int(np.ceil(len(cust)*.01)))).revenue.sum() / cust.revenue.sum())
profile["top_10pct_customer_revenue_share"] = float(cust.head(max(1, int(np.ceil(len(cust)*.10)))).revenue.sum() / cust.revenue.sum())

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(monthly.Month, monthly.revenue, marker="o", color="#2864dc")
ax.set(title="Monthly positive sales revenue", xlabel="Month", ylabel="Revenue (£)")
ax.tick_params(axis="x", rotation=45); fig.tight_layout(); fig.savefig(OUT / "fig1_monthly_revenue.png", dpi=180); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
topc = country.head(10).sort_values("revenue")
ax.barh(topc.Country, topc.revenue, color="#18a67e")
ax.set(title="Top ten countries by positive sales revenue", xlabel="Revenue (£)", ylabel="Country")
fig.tight_layout(); fig.savefig(OUT / "fig2_country_revenue.png", dpi=180); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
vals = np.log1p(clean.LineAmount.clip(lower=0))
sns.histplot(vals, bins=80, ax=ax, color="#7c4dff")
ax.set(title="Distribution of positive line value (log scale)", xlabel="log(1 + line value £)", ylabel="Line items")
fig.tight_layout(); fig.savefig(OUT / "fig3_line_value_distribution.png", dpi=180); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 7))
corr_cols = ["recency_days", "orders", "unique_products", "monetary", "units", "avg_order_value", "orders_30d", "spend_30d", "target"]
sns.heatmap(model_df[corr_cols].corr(method="spearman"), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
ax.set(title="Spearman relationships in customer snapshots")
fig.tight_layout(); fig.savefig(OUT / "fig4_spearman_heatmap.png", dpi=180); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
ss = snapshot_summary.copy(); ss["snapshot"] = ss.snapshot.dt.strftime("%Y-%m")
ax.bar(ss.snapshot, ss["count"] - ss["sum"], label="No purchase", color="#b7c0ce")
ax.bar(ss.snapshot, ss["sum"], bottom=ss["count"] - ss["sum"], label="Purchase", color="#f05a47")
ax.set(title="Thirty-day purchase target by snapshot", xlabel="Snapshot", ylabel="Customer snapshots")
ax.legend(); ax.tick_params(axis="x", rotation=45); fig.tight_layout(); fig.savefig(OUT / "fig5_target_imbalance.png", dpi=180); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
imp10 = importance.head(10).sort_values("importance_mean")
ax.barh(imp10.feature, imp10.importance_mean, xerr=imp10.importance_std, color="#ff9d2e")
ax.set(title="Boosting model permutation importance on out-of-time test", xlabel="Decrease in PR-AUC", ylabel="Feature")
fig.tight_layout(); fig.savefig(OUT / "fig6_permutation_importance.png", dpi=180); plt.close(fig)

with open(OUT / "profile.json", "w", encoding="utf-8") as f:
    json.dump(profile, f, indent=2)
model_df.to_csv(OUT / "customer_snapshots.csv", index=False)
print(json.dumps(profile, indent=2))
print("\nMETRICS\n", metrics.to_string(index=False))
print("\nIMPORTANCE\n", importance.head(12).to_string(index=False))
print("\nMONTHLY\n", monthly.to_string(index=False))
print("\nCOUNTRY\n", country.head(10).to_string(index=False))
print("\nSNAPSHOTS\n", snapshot_summary.to_string(index=False))
