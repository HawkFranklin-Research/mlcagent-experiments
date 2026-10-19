import json
import mlca_runtime
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns") or [])
group_col = protocol.get("group_column")
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop_set = set(targets) | set(excluded)
if group_col:
    drop_set.add(group_col)
feature_cols = [c for c in train.columns if c not in drop_set]
n = len(train)

excluded_feat = {}
numy = {}
for tgt in targets:
    if not pd.api.types.is_numeric_dtype(train[tgt]) or not train[tgt].notna().all():
        continue
    yv = train[tgt].astype(float).values
    numy[tgt] = yv
    cand = [f for f in feature_cols if pd.api.types.is_numeric_dtype(train[f])
            and train[f].notna().all() and (train[f] >= 0).all()]
    for f in cand:
        if np.array_equal(train[f].astype(float).values, yv):
            excluded_feat[f] = f"equals target {tgt}"
    keys = [f for f in cand if f not in excluded_feat]
    vs = {f: train[f].astype(float).values for f in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if np.array_equal(vs[a] + vs[b], yv):
                excluded_feat[a] = f"sum {a}+{b}={tgt}"
                excluded_feat[b] = f"sum {a}+{b}={tgt}"
    if len(keys) <= 15:
        for i, a in enumerate(keys):
            for j, b in enumerate(keys[i + 1:], i + 1):
                for c in keys[j + 1:]:
                    if np.array_equal(vs[a] + vs[b] + vs[c], yv):
                        for f in (a, b, c):
                            excluded_feat[f] = f"3-sum={tgt}"

feature_cols = [f for f in feature_cols if f not in excluded_feat]
if excluded_feat:
    print("excluded: " + "; ".join(f"{k}({v})" for k, v in excluded_feat.items()))

warned = []
for tgt, yv in numy.items():
    for f in feature_cols:
        if pd.api.types.is_numeric_dtype(train[f]):
            try:
                c = train[f].corr(train[tgt])
                if abs(c) >= 0.98:
                    warned.append(f)
            except Exception:
                pass
if warned:
    print(f"corr_warn>=0.98 kept: {sorted(set(warned))}")

ph_tokens = {"?", "-", "NA", "N/A", ""}
ph_cols = []
for f in feature_cols:
    if not pd.api.types.is_numeric_dtype(train[f]) and not pd.api.types.is_datetime64_any_dtype(train[f]):
        s = train[f].astype(str).str.strip()
        if s.isin(ph_tokens).any():
            ph_cols.append(f)
            train[f] = s.mask(s.isin(ph_tokens), np.nan)
print(f"placeholders: {ph_cols or 'none'}")

numeric_cols, cat_cols, drop_more = [], [], []
for f in feature_cols:
    x = train[f]
    if pd.api.types.is_datetime64_any_dtype(x):
        drop_more.append(f)
    elif pd.api.types.is_numeric_dtype(x):
        vals = set(x.dropna().unique().tolist())
        if vals == {0, 1}:
            numeric_cols.append(f)
        elif pd.api.types.is_integer_dtype(x) and x.nunique() <= 15 and x.min() >= 0:
            cat_cols.append(f)
        else:
            numeric_cols.append(f)
    else:
        nuniq = x.nunique(dropna=True)
        if nuniq > 5 and nuniq > 0.05 * n:
            drop_more.append(f)
        else:
            cat_cols.append(f)

est_w = len(numeric_cols) + sum(train[f].nunique() + 1 for f in cat_cols)
use_pd = est_w <= 300
print(f"num={len(numeric_cols)} cat={len(cat_cols)} dropped={drop_more} pd_out={use_pd} est_w={est_w}")

transformers = []
if numeric_cols:
    transformers.append(("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ]), numeric_cols))
if cat_cols:
    transformers.append(("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=not use_pd)),
    ]), cat_cols))

pre = ColumnTransformer(transformers, remainder="drop")
if use_pd:
    pre.set_output(transform="pandas")

X_train = train[numeric_cols + cat_cols].copy()
res = mlca_runtime.fit_record(pre, X_train, partition="train")
fitted = res if res is not None else pre
mlca_runtime.save_model(fitted, "preprocessor.pkl")
try:
    out = fitted.transform(X_train.head(3))
    print(f"saved preprocessor.pkl out_width={out.shape[1]}")
except Exception as e:
    print(f"saved preprocessor.pkl (probe failed: {type(e).__name__})")
