import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

mlca_runtime.set_seed(42)
rng = np.random.default_rng(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
task = proto["task"]
primary = task["primary_metric"]
targets = proto["targets"]
excluded = set(proto.get("excluded_columns", []))
group = proto.get("group_column")
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
test = pd.read_parquet(inp["test"])

def key_like(s):
    if s.dtype == object or str(s.dtype).startswith("string"):
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s):
        u = s.dropna().sort_values().to_numpy()
        return len(u) == 1 or np.all(np.diff(u) == 1)
    return False

feature_cols = [c for c in train.columns
                if c not in set(targets) | excluded and c != group and not key_like(train[c])]
X_test = test[feature_cols].copy()

def load_pkl(k):
    with open(inp[k], "rb") as f:
        return pickle.load(f)

vdf = pd.read_csv(inp["09_validate_score_validation_N1TQDQ22/validation_metrics.csv"])
m9 = json.loads(inp["09_validate_score_validation_N1TQDQ22/manifest.json"].read_text())

def find_threshold(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if "threshold" in str(k).lower() and isinstance(v, (int, float)) and 0 <= v <= 1:
                return float(v)
            r = find_threshold(v)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find_threshold(v)
            if r is not None:
                return r
    return None

threshold = find_threshold(m9)
if threshold is None and "threshold" in vdf.columns:
    t = pd.to_numeric(vdf["threshold"], errors="coerce").dropna()
    threshold = float(t.iloc[0]) if len(t) else None
threshold = 0.5 if threshold is None else threshold

def select_model():
    metric_col = next((c for c in vdf.columns if str(c).lower().replace(" ", "_") == primary.lower()), None)
    model_col = next((c for c in vdf.columns if "model" in str(c).lower() or "candidate" in str(c).lower()), None)
    if metric_col is not None:
        rows = vdf
        if "metric" in vdf.columns:
            rows = vdf[vdf["metric"].astype(str).str.lower() == primary.lower()]
        if len(rows) and model_col is not None:
            vals = pd.to_numeric(rows[metric_col], errors="coerce")
            if vals.notna().any():
                idx = vals.idxmax() if primary.lower() in {"roc_auc", "pr_auc", "balanced_accuracy", "f1", "f1_macro", "accuracy"} else vals.idxmin()
                name = str(rows.loc[idx, model_col]).lower()
                return "alternative" if ("alternative" in name or "08" in name or "alt" in name) else "main"
        if len(rows) == 1 and model_col is not None:
            name = str(rows.iloc[0][model_col]).lower()
            return "alternative" if ("alternative" in name or "08" in name or "alt" in name) else "main"
    s = json.dumps(m9).lower()
    return "alternative" if ("alternative" in s or "08_model_train_alternative" in s) else "main"

model_path = ("08_model_train_alternative_6PKR6EV7/alternative_model.pkl"
              if select_model() == "alternative" else
              "07_model_train_logistic_YKXH5QEQ/model.pkl")
model = load_pkl(model_path)

def model_cols(m):
    cols = getattr(m, "feature_names_in_", None)
    if cols is None and isinstance(m, Pipeline):
        first = m.named_steps[list(m.named_steps)[0]]
        cols = getattr(first, "feature_names_in_", None)
    return list(cols) if cols is not None else feature_cols

X_test = test[model_cols(model)].copy()
target = targets[0]
y_test = test[target].to_numpy()
proba = model.predict_proba(X_test)
classes = list(getattr(model, "classes_", []))
if not classes and isinstance(model, Pipeline):
    classes = list(model.named_steps[list(model.named_steps)[-1]].classes_)
pos = task.get("positive_label", task.get("positive_class"))
pos = classes[1] if pos is None or pos not in classes else pos
pos_idx = classes.index(pos)
other = classes[0] if pos == classes[1] else classes[1]
y_bin = (y_test == pos).astype(int)
p_pos = proba[:, pos_idx]
y_pred_label = np.where(p_pos >= threshold, pos, other)

def metrics(yy, pp):
    yp = (pp >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(yy, yp, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "roc_auc": roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan,
        "pr_auc": average_precision_score(yy, pp),
        "balanced_accuracy": (sens + spec) / 2,
        "brier": float(np.mean((pp - yy) ** 2)),
        "sensitivity": sens,
        "specificity": spec,
    }

vals = metrics(y_bin, p_pos)
boot = {k: [] for k in vals}
n = len(y_bin)
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yy, pp = y_bin[idx], p_pos[idx]
    if len(np.unique(yy)) < 2:
        continue
    bm = metrics(yy, pp)
    for k in boot:
        boot[k].append(bm[k])

rows = []
for k, v in vals.items():
    b = np.asarray(boot[k], dtype=float)
    lo, hi = (np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5)) if len(b) else (np.nan, np.nan)
    rows.append({"metric": k, "value": v, "ci_low": lo, "ci_high": hi})
mt = pd.DataFrame(rows)
mlca_runtime.save_table(mt, "test_metrics.csv", "Frozen model test metrics with bootstrap CI")
pv = mt.loc[mt["metric"] == primary].iloc[0]
mlca_runtime.record_metric(primary, float(pv["value"]), "test", ci=(float(pv["ci_low"]), float(pv["ci_high"])))
if primary.lower() == "roc_auc":
    mlca_runtime.record_fact_candidate("test_roc_auc", float(vals["roc_auc"]), source="test_metrics.csv")

pred = pd.DataFrame({"row_id": test.index, "partition": "test",
                     f"y_true__{target}": y_test, f"y_pred__{target}": y_pred_label})
for j, c in enumerate(classes):
    pred[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(pred, "test_predictions.parquet")
print(f"test rows={len(test)} model={model_path.split('/')[0]} threshold={threshold:.4g}")
print(f"{primary}={pv['value']:.4g} CI=({pv['ci_low']:.4g},{pv['ci_high']:.4g})")
