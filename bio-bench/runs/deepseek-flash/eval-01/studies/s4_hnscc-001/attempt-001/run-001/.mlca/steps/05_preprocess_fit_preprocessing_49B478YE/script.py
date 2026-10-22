import mlca_runtime
import json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing {key!r}; available top-level keys: {list(protocol)}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
task = protocol["task"]
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
outcome_cols = set(targets) | set(excluded)
if group:
    outcome_cols.add(group)
for source in (protocol, task):
    for key in ("event_column", "event_columns", "time_column", "time_columns"):
        v = source.get(key)
        if isinstance(v, str):
            outcome_cols.add(v)
        elif isinstance(v, (list, tuple)):
            outcome_cols.update(v)

feature_cols = [c for c in train.columns if c not in outcome_cols]
n_rows = len(train)
drop_reasons = {}

text_cols = [c for c in train.columns if c not in outcome_cols and (
    pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])
    or isinstance(train[c].dtype, pd.CategoricalDtype))]
placeholder_cols = []
placeholders = {"?", "-", "NA", "N/A"}
for c in text_cols:
    s = train[c].astype("string").str.strip()
    mask = s.isin(placeholders) | s.str.fullmatch(r"\s*").fillna(False)
    if mask.any():
        placeholder_cols.append(c)
        train[c] = s.mask(mask, pd.NA)
        vs = val[c].astype("string").str.strip()
        vmask = vs.isin(placeholders) | vs.str.fullmatch(r"\s*").fillna(False)
        val[c] = vs.mask(vmask, pd.NA)

for c in list(feature_cols):
    s = train[c]
    if s.isna().all():
        drop_reasons[c] = "all missing in train"
    elif pd.api.types.is_datetime64_any_dtype(s):
        dt_tr, dt_va = pd.to_datetime(train[c]), pd.to_datetime(val[c])
        for part in ("year", "month", "day", "dayofweek", "hour"):
            new = f"{c}_{part}"
            train[new] = getattr(dt_tr.dt, part)
            val[new] = getattr(dt_va.dt, part)
            feature_cols.append(new)
        drop_reasons[c] = "raw datetime replaced by parts"
    elif (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
          or isinstance(s.dtype, pd.CategoricalDtype)) and s.nunique(dropna=True) == n_rows:
        drop_reasons[c] = "text code all distinct"
    elif pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == n_rows:
        if any(tok in c.lower() for tok in ("id", "code", "key")):
            drop_reasons[c] = "integer identifier sequence"

numeric_candidates = [c for c in feature_cols if c not in drop_reasons
                      and pd.api.types.is_numeric_dtype(train[c])
                      and not pd.api.types.is_bool_dtype(train[c])]
for t in targets:
    if t not in train.columns or not pd.api.types.is_numeric_dtype(train[t]):
        continue
    y = train[t]
    for c in numeric_candidates:
        if c not in drop_reasons and train[c].equals(y):
            drop_reasons[c] = f"exact copy of {t}"
nonneg = [c for c in numeric_candidates if c not in drop_reasons
          and (train[c].dropna() >= 0).all()]
if 1 < len(nonneg) <= 60:
    for t in targets:
        if t not in train.columns or not pd.api.types.is_numeric_dtype(train[t]):
            continue
        y = train[t]
        for r in (2, 3):
            for combo in combinations(nonneg, r):
                if any(c in drop_reasons for c in combo):
                    continue
                sub = train[list(combo)]
                if sub.isna().any().any():
                    continue
                if np.allclose(sub.sum(axis=1).values, y.values):
                    for c in combo:
                        drop_reasons[c] = f"sum-to-{t}"

high_corr = set()
for t in targets:
    if t not in train.columns or not pd.api.types.is_numeric_dtype(train[t]):
        continue
    for c in numeric_candidates:
        if c in drop_reasons:
            continue
        corr = train[c].corr(train[t])
        if pd.notna(corr) and abs(corr) >= 0.98:
            high_corr.add(c)

final_cols, num_cols, cat_cols = [], [], []
for c in feature_cols:
    if c in drop_reasons:
        continue
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
        final_cols.append(c)
    else:
        if s.nunique(dropna=True) > 0.05 * n_rows:
            drop_reasons[c] = "high-cardinality text"
        else:
            cat_cols.append(c)
            final_cols.append(c)

cat_width = sum(train[c].nunique(dropna=True) + 1 for c in cat_cols)
wide = len(num_cols) + cat_width > 300
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                     ("encoder", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=not wide))])
transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
pre = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
if not wide:
    pre.set_output(transform="pandas")

X_train, X_val = train[final_cols].copy(), val[final_cols].copy()
pre = mlca_runtime.fit_record(pre, X_train, "train")
Xt = pre.transform(X_val)
mlca_runtime.save_model(pre, "preprocessor.pkl")

print(f"Features: {len(final_cols)} (numeric={len(num_cols)}, categorical={len(cat_cols)})")
print(f"Placeholders in {len(placeholder_cols)} text columns" + (f": {placeholder_cols[:5]}" if placeholder_cols else ""))
print("Excluded: " + ("; ".join(f"{k}={v}" for k, v in list(drop_reasons.items())[:8]) if drop_reasons else "none"))
print("High correlation warnings: " + (", ".join(sorted(high_corr)[:8]) if high_corr else "none"))
print(f"Transform val: shape={Xt.shape}, wide_sparse={wide}")
