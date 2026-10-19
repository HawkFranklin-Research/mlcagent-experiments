import json
import itertools
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def require(d, keys, label):
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"{label} missing {missing}; available: {sorted(d.keys())}")

require(protocol, ["targets", "excluded_columns", "group_column", "task"], "protocol")
require(protocol["task"], ["primary_metric"], "protocol.task")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]

train = pd.read_parquet(inputs["train"])
all_cols = list(train.columns)
base_excluded = set(targets) | set(excluded) | {group}
base_candidates = [c for c in all_cols if c not in base_excluded]

def is_textish(s):
    return (pd.api.types.is_object_dtype(s.dtype) or
            pd.api.types.is_string_dtype(s.dtype) or
            isinstance(s.dtype, pd.CategoricalDtype))

key_like = []
for c in base_candidates:
    s = train[c]
    if s.isna().sum() == 0 and s.nunique(dropna=True) == len(train):
        if is_textish(s):
            key_like.append(c)
        elif pd.api.types.is_integer_dtype(s) and any(t in c.lower() for t in ["id", "key", "code", "index"]):
            key_like.append(c)

candidate_features = [c for c in base_candidates if c not in key_like]

placeholder_tokens = {"?", "-", "NA", "N/A"}
placeholder_cols = []
for c in candidate_features:
    s = train[c]
    if is_textish(s):
        stripped = s.astype("string").str.strip()
        mask = stripped.notna() & (stripped.isin(placeholder_tokens) | stripped.eq(""))
        if mask.any():
            placeholder_cols.append(c)
            train[c] = stripped.mask(mask, pd.NA)

derivation_excluded = {}
for t in targets:
    for f in candidate_features:
        if f not in derivation_excluded and train[f].equals(train[t]):
            derivation_excluded[f] = f"exact copy of {t}"

for t in targets:
    if pd.api.types.is_numeric_dtype(train[t]):
        y = train[t].to_numpy(dtype=float)
        nums = [f for f in candidate_features if f not in derivation_excluded
                and pd.api.types.is_numeric_dtype(train[f])
                and (train[f] >= 0).all()]
        if len(nums) <= 30:
            for r in (2, 3):
                for combo in itertools.combinations(nums, r):
                    if np.allclose(train[list(combo)].sum(axis=1).to_numpy(dtype=float), y):
                        for f in combo:
                            derivation_excluded[f] = f"sum to {t}"

corr_warn = []
for t in targets:
    if pd.api.types.is_numeric_dtype(train[t]):
        for f in candidate_features:
            if f in derivation_excluded or not pd.api.types.is_numeric_dtype(train[f]):
                continue
            corr = train[[f, t]].corr().iloc[0, 1]
            if abs(corr) >= 0.98:
                corr_warn.append(f)

working = [f for f in candidate_features if f not in derivation_excluded]
date_cols = []
for c in working:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        date_cols.append(c)
    elif is_textish(s) and s.notna().any():
        sample = s.dropna().astype(str).head(30)
        if sample.str.contains(r"[-/:T]", regex=True).mean() >= 0.8:
            parsed = pd.to_datetime(s, errors="coerce")
            if parsed.notna().mean() >= 0.8:
                date_cols.append(c)

derived_date_cols = []
for c in date_cols:
    dt = pd.to_datetime(train[c], errors="coerce")
    parts = {"year": dt.dt.year, "month": dt.dt.month, "day": dt.dt.day,
             "dow": dt.dt.dayofweek, "hour": dt.dt.hour, "minute": dt.dt.minute}
    for name, vals in parts.items():
        new = f"{c}_{name}"
        train[new] = vals
        derived_date_cols.append(new)
working = [f for f in working if f not in date_cols] + derived_date_cols

num_cols, cat_cols, high_card_dropped = [], [], []
threshold = max(1, int(0.05 * len(train)))
for c in working:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s.dtype):
        num_cols.append(c)
    else:
        if s.nunique(dropna=True) > threshold:
            high_card_dropped.append(c)
        else:
            cat_cols.append(c)

final_features = num_cols + cat_cols
bad = set(final_features) & (set(targets) | set(excluded) | {group})
if bad:
    raise ValueError(f"target/excluded/group leaked into features: {sorted(bad)}")

width = len(num_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
dense = width <= 300
transformers = []
if num_cols:
    num_pipe = mlca_runtime.Pipeline if hasattr(mlca_runtime, "Pipeline") else None
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    if dense:
        num_pipe.set_output(transform="pandas")
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
    ])
    if dense:
        cat_pipe.set_output(transform="pandas")
    transformers.append(("cat", cat_pipe, cat_cols))

from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(transformers, remainder="drop")
if dense:
    preprocessor.set_output(transform="pandas")

X_train = train[final_features]
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, partition="train")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

lines = [
    f"train rows={len(train)} cols={train.shape[1]}",
    f"features numeric={len(num_cols)} cat={len(cat_cols)} high_card_dropped={len(high_card_dropped)}",
]
if placeholder_cols:
    lines.append("placeholders: " + ", ".join(placeholder_cols[:10]))
if derivation_excluded:
    lines.append("derivation excluded: " + ", ".join(f"{k} ({v})" for k, v in list(derivation_excluded.items())[:10]))
if corr_warn:
    lines.append("corr>=0.98 kept: " + ", ".join(corr_warn[:10]))
lines.append(f"saved preprocessor.pkl width={width}")
for line in lines[:5]:
    print(line)
