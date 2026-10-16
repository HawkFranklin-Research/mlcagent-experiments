import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)
drop = set(targets) | set(excluded)
if group:
    drop.add(group)
candidates = [c for c in train.columns if c not in drop]

def is_keylike(s):
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_integer_dtype(s):
        return False
    if pd.api.types.is_integer_dtype(s) or s.dtype == object:
        return s.nunique(dropna=True) == len(s) and s.nunique(dropna=True) > 1
    return False

features = [c for c in candidates if not is_keylike(train[c])]
n = len(train)
num_cols, low_cat, high_cat, date_cols = [], [], [], []
for c in features:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    elif pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    else:
        try:
            parsed = pd.to_datetime(s.dropna().head(20), errors="coerce")
            if parsed.notna().mean() > 0.8:
                date_cols.append(c)
                continue
        except Exception:
            pass
        nun = s.nunique(dropna=True)
        (low_cat if nun <= max(1, int(0.05 * n)) else high_cat).append(c)

ph_tokens = {"?", "-", "NA", "N/A", ""}
ph_cols = [c for c in low_cat + high_cat
           if train[c].dropna().astype(str).str.strip().isin(ph_tokens).any()]
print(f"placeholder tokens in {len(ph_cols)} columns")

log_cols = [c for c in num_cols
            if len(train[c].dropna()) > 0 and (train[c].dropna() >= 0).all()
            and train[c].dropna().skew() > 2]

def clean_text(X, cols):
    X = X.copy()
    for c in cols:
        s = X[c].astype(str).str.strip()
        X[c] = s.where(~s.isin(ph_tokens), np.nan)
    return X

def log1p_cols(X, cols):
    X = X.copy()
    for c in cols:
        X[c] = np.log1p(X[c].clip(lower=0))
    return X

class DateFeat(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        out = {}
        for c in self.cols:
            d = pd.to_datetime(X[c], errors="coerce")
            out[c + "_hour"] = d.dt.hour
            out[c + "_dow"] = d.dt.dayofweek
            out[c + "_month"] = d.dt.month
        return pd.DataFrame(out, index=X.index)
    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_{p}" for c in self.cols for p in ["hour", "dow", "month"]])

width = len(num_cols) + 3 * len(date_cols) + sum(train[c].nunique(dropna=True) + 1 for c in low_cat)
use_pandas = width <= 300

transformers = []
if num_cols:
    steps = []
    if log_cols:
        steps.append(("log", FunctionTransformer(log1p_cols, kw_args={"cols": log_cols})))
    steps += [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    transformers.append(("num", Pipeline(steps), num_cols))
if low_cat:
    transformers.append(("cat", Pipeline([
        ("clean", FunctionTransformer(clean_text, kw_args={"cols": low_cat})),
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=not use_pandas)),
    ]), low_cat))
if date_cols:
    transformers.append(("date", Pipeline([
        ("dt", DateFeat(date_cols)),
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ]), date_cols))

ct = ColumnTransformer(transformers, remainder="drop")
if use_pandas:
    ct.set_output(transform="pandas")
    for _, t, _ in transformers:
        if isinstance(t, Pipeline):
            t.set_output(transform="pandas")

mlca_runtime.fit_record(ct, train[features], partition="train")
mlca_runtime.save_model(ct, "preprocessor.pkl")
print(f"features={len(features)} width={width} pandas={use_pandas}")
