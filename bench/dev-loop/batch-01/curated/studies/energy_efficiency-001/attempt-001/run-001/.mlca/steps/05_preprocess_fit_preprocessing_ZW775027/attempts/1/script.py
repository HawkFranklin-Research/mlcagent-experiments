import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
for k in ("targets", "excluded_columns"):
    if k not in proto:
        raise KeyError(f"Missing {k}; available: {list(proto.keys())}")
targets = list(proto["targets"])
excluded = list(proto.get("excluded_columns") or [])
group = proto.get("group_column")

train = pd.read_parquet(inputs["train"])
drop = set(targets) | set(excluded) | ({group} if group else set())

feat_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s):
        continue
    feat_cols.append(c)

excluded_feats = {}
num_feats = [c for c in feat_cols if pd.api.types.is_numeric_dtype(train[c])]
for t in targets:
    tv = train[t].to_numpy()
    for c in num_feats:
        if np.array_equal(train[c].to_numpy(), tv):
            excluded_feats[c] = f"equals {t}"
    nonneg = [c for c in num_feats if (train[c] >= 0).all()]
    for i, a in enumerate(nonneg):
        for j in range(i + 1, len(nonneg)):
            b = nonneg[j]
            if np.allclose(train[a].to_numpy() + train[b].to_numpy(), tv):
                excluded_feats.setdefault(a, f"sum with {b} = {t}")
                excluded_feats.setdefault(b, f"sum with {a} = {t}")
            for k in range(j + 1, len(nonneg)):
                c2 = nonneg[k]
                if np.allclose(train[a].to_numpy() + train[b].to_numpy() + train[c2].to_numpy(), tv):
                    for x in (a, b, c2):
                        excluded_feats.setdefault(x, f"sum = {t}")

if excluded_feats:
    print("excluded: " + "; ".join(f"{c} ({r})" for c, r in excluded_feats.items()))
feat_cols = [c for c in feat_cols if c not in excluded_feats]

warn = []
for t in targets:
    for c in feat_cols:
        if pd.api.types.is_numeric_dtype(train[c]):
            r = abs(train[c].corr(train[t]))
            if r >= 0.98:
                warn.append(f"{c}({r:.2f})")
if warn:
    print("high-corr kept: " + ", ".join(warn))

tokens = {"?", "-", "NA", "N/A"}
ph_cols = []
for c in feat_cols:
    s = train[c]
    if not pd.api.types.is_numeric_dtype(s):
        v = s.dropna().astype(str).str.strip()
        if v.isin(tokens).any() or (v == "").any():
            ph_cols.append(c)
if ph_cols:
    print(f"placeholders in {len(ph_cols)} cols: {ph_cols}")

n = len(train)
num_cols, low_cat, high_cat, date_cols = [], [], [], []
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
        continue
    try:
        pd.to_datetime(s.dropna().head(20), errors="raise")
        date_cols.append(c)
        continue
    except Exception:
        pass
    nu = s.nunique(dropna=True)
    if nu <= max(1, int(0.05 * n)):
        low_cat.append(c)
    else:
        high_cat.append(c)

class TextClean(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in X.columns:
            s = X[c].astype(str).str.strip()
            X[c] = s.mask(s.isin(tokens) | (s == ""), np.nan)
        return X

class DateFeat(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = pd.DataFrame(X)
        out = {}
        for c in X.columns:
            d = pd.to_datetime(X[c], errors="coerce")
            out[c + "_hour"] = d.dt.hour
            out[c + "_dow"] = d.dt.dayofweek
            out[c + "_month"] = d.dt.month
        return pd.DataFrame(out, index=X.index)

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat_pipe = Pipeline([("clean", TextClean()), ("imp", SimpleImputer(strategy="most_frequent")),
                     ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
date_pipe = Pipeline([("dt", DateFeat()), ("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])

transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if low_cat:
    transformers.append(("cat", cat_pipe, low_cat))
if date_cols:
    transformers.append(("date", date_pipe, date_cols))

ct = ColumnTransformer(transformers, remainder="drop")
ct.set_output(transform="pandas")
mlca_runtime.fit_record(ct, train[feat_cols], "train")
width = ct.transform(train[feat_cols].head(5)).shape[1]
mlca_runtime.save_model(ct, "preprocessor.pkl")
print(f"features={len(feat_cols)} width={width}")
