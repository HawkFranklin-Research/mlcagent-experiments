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
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
for key in ("targets", "excluded_columns"):
    if key not in proto:
        raise KeyError(f"Missing {key}; available top-level keys: {list(proto)}")
targets = list(proto["targets"])
excluded = list(proto.get("excluded_columns", []))
group = proto.get("group_column")
train = pd.read_parquet(inp["train"])

missing_targets = [t for t in targets if t not in train.columns]
if missing_targets:
    raise KeyError(f"Missing target columns: {missing_targets}")

drop_cols = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in drop_cols]

def key_like(s):
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=False) == len(s):
        vals = s.dropna().sort_values()
        return len(vals) == len(s) and np.array_equal(vals.to_numpy(), np.arange(vals.iloc[0], vals.iloc[0] + len(vals)))
    return False

key_like_cols = [c for c in feature_cols if key_like(train[c])]
if key_like_cols:
    print("drop_key_like=" + ",".join(key_like_cols))
feature_cols = [c for c in feature_cols if c not in key_like_cols]

ph_tokens = ["?", "-", "NA", "N/A", ""]
ph_cols = 0
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        ss = s.astype("string").fillna("").str.strip()
        if ss.isin(ph_tokens).any():
            ph_cols += 1
print(f"placeholder_columns={ph_cols}")

numeric_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
derived = {}
for t in targets:
    y = train[t]
    for c in feature_cols:
        s = train[c]
        if c == t:
            continue
        if pd.api.types.is_numeric_dtype(s.dtype) and pd.api.types.is_numeric_dtype(y.dtype):
            if s.notna().all() and y.notna().all() and np.allclose(s.to_numpy(float), y.to_numpy(float)):
                derived[c] = f"equals {t}"
        elif s.equals(y):
            derived[c] = f"equals {t}"
    if pd.api.types.is_numeric_dtype(y.dtype):
        cand = [c for c in numeric_feats if c not in derived and train[c].notna().all() and (train[c] >= 0).all()]
        for r in (1, 2, 3):
            for combo in combinations(cand, r):
                total = train[list(combo)].sum(axis=1)
                if y.notna().all() and np.allclose(total.to_numpy(float), y.to_numpy(float)):
                    for c in combo:
                        derived[c] = f"sum={t}"
if derived:
    print("excluded_derived=" + ";".join(f"{c}:{r}" for c, r in derived.items()))
feature_cols = [c for c in feature_cols if c not in derived]

warn = []
for t in targets:
    if not pd.api.types.is_numeric_dtype(train[t].dtype):
        continue
    for c in feature_cols:
        if c in numeric_feats and train[c].notna().all() and train[t].notna().all():
            corr = np.corrcoef(train[c].to_numpy(float), train[t].to_numpy(float))[0, 1]
            if abs(corr) >= 0.98:
                warn.append(f"{c}~{t}({corr:.3f})")
if warn:
    print("corr_warn=" + ",".join(warn))

numeric_cols, cat_cols, date_cols, dropped = [], [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s.dtype):
        numeric_cols.append(c)
    else:
        if s.nunique(dropna=True) > 0.05 * len(train):
            dropped.append(c)
        else:
            cat_cols.append(c)
if dropped:
    print("drop_high_card=" + ",".join(dropped))
if date_cols:
    print("date_cols=" + ",".join(date_cols))
feature_cols = numeric_cols + cat_cols
width = len(numeric_cols) + sum(train[c].nunique(dropna=False) for c in cat_cols)

if width <= 300:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    num_pipe.set_output(transform="pandas")
    transformers = [("num", num_pipe, numeric_cols)]
    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        cat_pipe.set_output(transform="pandas")
        transformers.append(("cat", cat_pipe, cat_cols))
    pre = ColumnTransformer(transformers, remainder="drop")
    pre.set_output(transform="pandas")
else:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    transformers = [("num", num_pipe, numeric_cols)]
    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))
    pre = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)

mlca_runtime.fit_record(pre, train[feature_cols], "train")
mlca_runtime.save_model(pre, "preprocessor.pkl")
mlca_runtime.record_metric("n_features_used", float(len(feature_cols)), "train")
print(f"used_features={len(feature_cols)} numeric={len(numeric_cols)} categorical={len(cat_cols)}")
