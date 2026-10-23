import mlca_runtime
import json
import itertools
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
protocol = json.loads(inp["_protocol"].read_text())

for k in ["targets", "excluded_columns", "group_column", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing protocol key {k!r}; available keys: {list(protocol.keys())}")
targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group = protocol["group_column"]
primary_metric = protocol["task"]["primary_metric"]

msg = []
placeholders = {"?", "-", "NA", "N/A"}
placeholder_cols = set()
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    if pd.api.types.is_string_dtype(train[c]) or pd.api.types.is_object_dtype(train[c]):
        s_tr = train[c].astype("string").str.strip()
        s_va = val[c].astype("string").str.strip()
        mask_tr = s_tr.isin(placeholders) | (s_tr == "")
        if mask_tr.any():
            train[c] = s_tr.mask(mask_tr, np.nan)
            val[c] = s_va.mask(s_va.isin(placeholders) | (s_va == ""), np.nan)
            placeholder_cols.add(c)
msg.append(f"Placeholder tokens in {len(placeholder_cols)} text columns" if placeholder_cols else "Placeholder tokens: none")

for c in list(train.columns):
    if c in targets or c in excluded or c == group:
        continue
    is_dt = pd.api.types.is_datetime64_any_dtype(train[c])
    is_dt_name = pd.api.types.is_string_dtype(train[c]) and any(x in c.lower() for x in ["date", "time"])
    if is_dt or is_dt_name:
        dt_tr = pd.to_datetime(train[c], errors="coerce")
        dt_va = pd.to_datetime(val[c], errors="coerce")
        for part, attr in [("month", "month"), ("dow", "dayofweek"), ("hour", "hour")]:
            train[f"{c}_{part}"] = getattr(dt_tr.dt, attr)
            val[f"{c}_{part}"] = getattr(dt_va.dt, attr)
        train = train.drop(columns=[c])
        val = val.drop(columns=[c])

for c in train.columns:
    if pd.api.types.is_bool_dtype(train[c]):
        train[c] = train[c].astype("int8")
        val[c] = val[c].astype("int8")

key_like = set()
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    if pd.api.types.is_string_dtype(train[c]) or pd.api.types.is_object_dtype(train[c]):
        if train[c].nunique(dropna=True) == len(train):
            key_like.add(c)
    if pd.api.types.is_integer_dtype(train[c]):
        vals = train[c].dropna()
        if len(vals) == len(train) and vals.max() - vals.min() + 1 == len(train):
            key_like.add(c)

candidate_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group and c not in key_like]
derived = set()
for t in targets:
    y = train[t]
    for f in candidate_cols:
        if train[f].equals(y):
            derived.add(f)
        elif pd.api.types.is_numeric_dtype(train[f]) and pd.api.types.is_numeric_dtype(y):
            if np.array_equal(train[f].values, y.values, equal_nan=True):
                derived.add(f)
    if pd.api.types.is_numeric_dtype(y):
        nonneg = [f for f in candidate_cols if pd.api.types.is_numeric_dtype(train[f]) and (train[f] >= 0).all()]
        for r in range(1, 4):
            for combo in itertools.combinations(nonneg, r):
                s = train[list(combo)].sum(axis=1).astype(float).values
                if np.array_equal(s, y.astype(float).values, equal_nan=True):
                    derived.update(combo)
if derived:
    msg.append("Excluded derived features: " + ", ".join(sorted(derived)))

feature_cols = [c for c in candidate_cols if c not in derived]
corr_warn = []
for t in targets:
    if not pd.api.types.is_numeric_dtype(train[t]):
        continue
    for f in feature_cols:
        if pd.api.types.is_numeric_dtype(train[f]):
            corr = train[f].corr(train[t])
            if pd.notna(corr) and abs(corr) >= 0.98:
                corr_warn.append(f)
if corr_warn:
    msg.append("High correlation with target: " + ", ".join(sorted(set(corr_warn))))

numeric_cols, categorical_cols, dropped = [], [], []
threshold = max(1, int(0.05 * len(train)))
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(train[c]):
        numeric_cols.append(c)
    elif train[c].nunique(dropna=True) <= threshold:
        categorical_cols.append(c)
    else:
        dropped.append(c)

feature_cols = numeric_cols + categorical_cols
if not feature_cols:
    raise ValueError("No usable features after exclusions")
X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()
y_train = train[targets[0]] if targets else None

est_width = len(numeric_cols) + sum(train[c].nunique(dropna=True) for c in categorical_cols)
use_pandas = est_width <= 300
transformers = []
if numeric_cols:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    transformers.append(("num", num_pipe, numeric_cols))
if categorical_cols:
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not use_pandas)),
    ])
    transformers.append(("cat", cat_pipe, categorical_cols))
ct = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0 if use_pandas else 0.3)
if use_pandas:
    ct.set_output(transform="pandas")

mlca_runtime.fit_record(ct, X_train, "train", y=y_train)
trans_val = ct.transform(X_val)
mlca_runtime.record_metric("n_features", float(len(feature_cols)), split="train")
mlca_runtime.record_metric("encoded_width", float(est_width), split="train")
mlca_runtime.save_model(ct, "preprocessor.pkl")

msg.append(f"Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical; dropped {len(dropped)} high-card/key-like")
msg.append(f"Encoded width {est_width}; val shape {trans_val.shape}")
for line in msg[:5]:
    print(line)
