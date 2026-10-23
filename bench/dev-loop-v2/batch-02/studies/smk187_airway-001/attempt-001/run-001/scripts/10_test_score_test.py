import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
try:
    targets = protocol["targets"]
    task = protocol["task"]
    excluded = protocol["excluded_columns"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}. Available top-level keys: {list(protocol.keys())}")
primary_metric = task["primary_metric"]
group = protocol.get("group_column", None)
train = pd.read_parquet(inputs["train"])
test = pd.read_parquet(inputs["test"])

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    s = train[c]
    if (s.dtype == object or pd.api.types.is_string_dtype(s.dtype)) and s.nunique(dropna=True) == s.notna().sum():
        continue
    feature_cols.append(c)

def find_threshold(inputs):
    for key in ["09_validate_score_validation_ZRCFGQ5B/validation_metrics.parquet",
                "09_validate_score_validation_ZRCFGQ5B/validation_metrics.csv"]:
        p = inputs.get(key)
        if p is None or not p.exists():
            continue
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        for c in df.columns:
            if "thresh" in c.lower():
                vals = pd.to_numeric(df[c], errors="coerce").dropna()
                if len(vals):
                    return float(vals.iloc[0])
        if "metric" in df.columns and "value" in df.columns:
            m = df["metric"].astype(str).str.lower().str.contains("thresh")
            if m.any():
                vals = pd.to_numeric(df.loc[m, "value"], errors="coerce").dropna()
                if len(vals):
                    return float(vals.iloc[0])
    return 0.5

with open(inputs["05_preprocess_fit_preprocessing_4ZKXP498/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_E9CTG9EJ/selector.pkl"], "rb") as f:
    sel = pickle.load(f)
with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

def accepts_raw(obj):
    return hasattr(obj, "feature_names_in_") and set(obj.feature_names_in_) == set(feature_cols)

if accepts_raw(model):
    full = model
elif accepts_raw(sel):
    full = Pipeline([("selector", sel), ("model", model)])
elif accepts_raw(pre):
    try:
        sel_in = set(sel.feature_names_in_)
    except Exception:
        sel_in = set()
    try:
        pre_out = set(pre.get_feature_names_out())
    except Exception:
        pre_out = set()
    if sel_in and sel_in == pre_out:
        full = Pipeline([("preprocessor", pre), ("selector", sel), ("model", model)])
    else:
        full = Pipeline([("preprocessor", pre), ("model", model)])
else:
    full = model

target = targets[0]
y_true = test[target].copy()
X = test[feature_cols].copy()
if hasattr(full, "feature_names_in_"):
    X = X[list(full.feature_names_in_)]
proba = full.predict_proba(X)
classes = np.asarray(full.classes_)
pos_label = task.get("positive_label", None)
if pos_label is None:
    pos_label = classes[1]
pos_idx = int(np.where(classes == pos_label)[0][0])
proba_pos = proba[:, pos_idx]
y_bin = (y_true == pos_label).astype(int).to_numpy()
threshold = find_threshold(inputs)
neg_label = [c for c in classes if c != pos_label][0]
y_pred_bin = (proba_pos >= threshold).astype(int)
y_pred_label = np.where(y_pred_bin == 1, pos_label, neg_label)

def calc(yb, pp, thr):
    yp = (pp >= thr).astype(int)
    out = {}
    try:
        out["roc_auc"] = roc_auc_score(yb, pp) if len(np.unique(yb)) == 2 else np.nan
    except Exception:
        out["roc_auc"] = np.nan
    try:
        out["pr_auc"] = average_precision_score(yb, pp) if yb.sum() > 0 else np.nan
    except Exception:
        out["pr_auc"] = np.nan
    try:
        out["balanced_accuracy"] = balanced_accuracy_score(yb, yp) if len(np.unique(yb)) == 2 else np.nan
    except Exception:
        out["balanced_accuracy"] = np.nan
    out["brier_score"] = float(np.mean((pp - yb) ** 2))
    tp = np.sum((yb == 1) & (yp == 1))
    fn = np.sum((yb == 1) & (yp == 0))
    tn = np.sum((yb == 0) & (yp == 0))
    fp = np.sum((yb == 0) & (yp == 1))
    out["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    out["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return out

base = calc(y_bin, proba_pos, threshold)
n = len(y_bin)
rng = np.random.default_rng(42)
boot = {k: [] for k in base}
for _ in range(1000):
    idx = rng.integers(0, n, n)
    vals = calc(y_bin[idx], proba_pos[idx], threshold)
    for k in boot:
        boot[k].append(vals[k])

rows = []
for k, v in base.items():
    arr = np.asarray(boot[k], dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr):
        lo, hi = np.percentile(arr, [2.5, 97.5])
    else:
        lo = hi = np.nan
    rows.append({"metric": k, "value": v, "ci_low": lo, "ci_high": hi})
metrics_df = pd.DataFrame(rows)

def sig4(x):
    return float(f"{x:.4g}") if pd.notna(x) else np.nan

for c in ["value", "ci_low", "ci_high"]:
    metrics_df[c] = metrics_df[c].map(sig4)
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

pred_df = pd.DataFrame({
    "row_id": test.index,
    "partition": "test",
    f"y_true__{target}": y_true.to_numpy(),
    f"y_pred__{target}": y_pred_label,
})
for j, c in enumerate(classes):
    pred_df[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(pred_df, "test_predictions.parquet")

pval = float(metrics_df.loc[metrics_df["metric"] == primary_metric, "value"].iloc[0])
plo = float(metrics_df.loc[metrics_df["metric"] == primary_metric, "ci_low"].iloc[0])
phi = float(metrics_df.loc[metrics_df["metric"] == primary_metric, "ci_high"].iloc[0])
mlca_runtime.record_metric(primary_metric, pval, "test", ci=(plo, phi))
if primary_metric == "roc_auc":
    mlca_runtime.record_fact_candidate("test_roc_auc", pval, source="test_metrics.csv")

print("test rows:", len(test))
print(f"{primary_metric}: {pval:.4g} CI [{plo:.4g}, {phi:.4g}]")
print("saved test_metrics.csv and test_predictions.parquet")
