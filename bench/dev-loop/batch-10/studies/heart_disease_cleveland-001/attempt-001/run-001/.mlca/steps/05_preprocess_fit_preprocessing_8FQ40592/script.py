import json
import numpy as np
import pandas as pd
import mlca_runtime
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(d, key):
    if key not in d:
        raise KeyError(f"Missing {key}; available top-level keys: {list(d.keys())}")
    return d[key]

targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group = need(protocol, "group_column")
task = need(protocol, "task")
need(task, "primary_metric")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def key_like(s, name):
    n = len(s)
    if s.nunique(dropna=False) != n:
        return False
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        return True
    if pd.api.types.is_integer_dtype(s):
        vals = s.dropna().sort_values().to_numpy()
        seq = len(vals) == n and np.array_equal(vals, np.arange(vals[0], vals[0] + n))
        id_like = bool(pd.Series([name]).str.contains(r"id|key|index|row|patient|subject|record", case=False).iloc[0])
        return bool(seq or id_like)
    return False

candidate_cols = [c for c in train.columns
                  if c not in set(targets + excluded + [group]) and not key_like(train[c], c)]

def eq_series(a, b):
    return bool(((a.isna() & b.isna()) | a.eq(b)).all())

derived = {}
for t in targets:
    y = train[t]
    for c in candidate_cols:
        if eq_series(train[c], y):
            derived[c] = f"equals {t}"
    nn = [c for c in candidate_cols
          if pd.api.types.is_numeric_dtype(train[c]) and train[c].notna().all() and (train[c] >= 0).all()]
    for r in (1, 2, 3):
        for cols in combinations(nn, r):
            s = train[list(cols)].sum(axis=1)
            if eq_series(s, y):
                for c in cols:
                    derived.setdefault(c, f"sum equals {t}")

feature_cols = [c for c in candidate_cols if c not in derived]
if derived:
    print("excluded derivations: " + "; ".join(f"{c} ({derived[c]})" for c in derived))
else:
    print("excluded derivations: none")

warn = []
if len(targets) == 1 and pd.api.types.is_numeric_dtype(train[targets[0]]):
    y = train[targets[0]]
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train[c]):
            corr = train[c].corr(y)
            if pd.notna(corr) and abs(corr) >= 0.98:
                warn.append(c)
if warn:
    print("corr>=0.98 kept: " + ", ".join(warn))

for c in feature_cols:
    if pd.api.types.is_datetime64_any_dtype(train[c]):
        train[c] = train[c].astype("int64") // 10**9
        val[c] = val[c].astype("int64") // 10**9

tokens = ["?", "-", "NA", "N/A"]
ph = {}
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        vals = s.dropna().astype(str)
        found = {tok for tok in tokens if (vals == tok).any()}
        ws = vals[vals.str.strip() == ""]
        if len(ws):
            found.update(ws.unique())
        if found:
            ph[c] = found
if ph:
    print(f"placeholders in {len(ph)} cols: {sorted(set().union(*ph.values()))}")
else:
    print("placeholders: none")

num_cols, cat_cols, high = [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    else:
        nun = s.nunique(dropna=True)
        if nun <= max(20, int(0.05 * len(train))):
            cat_cols.append(c)
        else:
            high.append(c)
if high:
    print("dropped high-card: " + ", ".join(high))

est_width = len(num_cols) + sum(train[c].nunique(dropna=True) + 1 for c in cat_cols)
wide = est_width > 300
transformers = []
for c in num_cols:
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler())])
    if not wide:
        pipe.set_output(transform="pandas")
    transformers.append((f"num_{c}", pipe, [c]))
for c in cat_cols:
    steps = []
    for i, tok in enumerate(sorted(ph.get(c, []))):
        steps.append((f"ph{i}", SimpleImputer(strategy="most_frequent", missing_values=tok)))
    steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
    steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not wide)))
    pipe = Pipeline(steps)
    if not wide:
        pipe.set_output(transform="pandas")
    transformers.append((f"cat_{c}", pipe, [c]))

preprocessor = ColumnTransformer(transformers, remainder="drop")
if not wide:
    preprocessor.set_output(transform="pandas")

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()
y_train = train[targets[0]] if len(targets) == 1 else train[targets]
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, "train", y_train)
Xt_val = preprocessor.transform(X_val)
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_encoded_features", float(Xt_val.shape[1]), "val")
print(f"features={len(feature_cols)} encoded={Xt_val.shape[1]} val_rows={Xt_val.shape[0]}")
