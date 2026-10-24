import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lifelines import CoxPHFitter

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()

protocol = json.loads(paths["_protocol"].read_text())
train = pd.read_parquet(paths["train"])
pre = pickle.loads(paths["05_preprocess_fit_preprocessing_M6JRF55T/preprocessor.pkl"].read_bytes())
_manifest = json.loads(paths["05_preprocess_fit_preprocessing_M6JRF55T/manifest.json"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing key {k!r}; available top-level keys: {list(d.keys())}")
    return d[k]

targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group_col = need(protocol, "group_column")
task = need(protocol, "task")
drop_cols = set(targets) | set(excluded) | {group_col}

feature_cols = [c for c in train.columns if c not in drop_cols]

def is_key_like(s):
    if s.nunique(dropna=False) != len(s):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        v = s.dropna()
        if len(v) == len(s) and v.is_monotonic_increasing:
            d = v.diff().dropna()
            return len(d) > 0 and (d == 1).all()
    return False

feature_cols = [c for c in feature_cols if not is_key_like(train[c])]
expected_cols = list(pre.feature_names_in_) if hasattr(pre, "feature_names_in_") else feature_cols
overlap = set(expected_cols) & drop_cols
if overlap:
    raise ValueError(f"Preprocessor expects excluded/target columns: {sorted(overlap)}")

X_train_all = train[expected_cols].copy()
duration_col, event_col = targets[0], targets[1]
duration = pd.to_numeric(train[duration_col], errors="coerce")
ev = train[event_col]
if pd.api.types.is_numeric_dtype(ev):
    event = (pd.to_numeric(ev, errors="coerce") > 0).astype(float)
else:
    event = ev.astype(str).str.lower().str.contains("dead|deceased|death|1", regex=True).astype(float)

valid = duration.notna() & event.notna() & (duration > 0)
X_valid_raw = train.loc[valid, expected_cols]
transformed = pre.transform(X_valid_raw)
if sparse.issparse(transformed):
    n_features = transformed.shape[1]
else:
    transformed = np.asarray(transformed)
    n_features = transformed.shape[1]

duration_valid = duration.loc[valid].to_numpy()
event_valid = event.loc[valid].to_numpy()
pvals = []
for j in range(n_features):
    if sparse.issparse(transformed):
        arr = transformed[:, j].toarray().ravel()
    else:
        arr = transformed[:, j].ravel()
    df = pd.DataFrame(
        {"duration": duration_valid, "event": event_valid, "x": arr},
        index=X_valid_raw.index,
    )
    cph = CoxPHFitter()
    try:
        fitted = mlca_runtime.fit_record(
            cph, df, "train", duration_col="duration", event_col="event"
        )
        p = float(fitted.summary.loc["x", "p"])
    except Exception:
        p = 1.0
    pvals.append((p, j))

selected_pairs = sorted(
    [x for x in pvals if np.isfinite(x[0]) and x[0] < 0.10], key=lambda x: x[0]
)[:20]
selected_indices = [j for _, j in selected_pairs]

if selected_indices:
    selector = ColumnTransformer([("keep", "passthrough", selected_indices)])
elif n_features:
    selector = ColumnTransformer([("drop", "drop", list(range(n_features)))])
else:
    selector = "passthrough"

final_pipeline = Pipeline([("preprocessing", pre), ("selector", selector)])
mlca_runtime.fit_record(final_pipeline, X_train_all, "train")
mlca_runtime.save_model(final_pipeline, "selector.pkl")
mlca_runtime.record_metric("n_selected_covariates", float(len(selected_indices)), "train")

print("train shape:", train.shape, "expected features:", len(expected_cols))
print("transformed features:", n_features)
print("selected covariates:", len(selected_indices))
