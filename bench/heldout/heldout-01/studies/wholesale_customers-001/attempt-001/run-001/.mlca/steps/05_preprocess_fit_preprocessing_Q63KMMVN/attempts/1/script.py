import mlca_runtime
import json
import itertools
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in proto]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {list(proto.keys())}")
targets = list(proto["targets"] or [])
excluded = list(proto["excluded_columns"] or [])
group_col = proto["group_column"]
task = proto["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing task['primary_metric']; task keys: {list(task.keys())}")

train = pd.read_parquet(inputs["train"])

placeholders = {"?", "-", "NA", "N/A", ""}
found = {}
for c in train.columns:
    if train[c].dtype.kind in "OUS":
        s = train[c].astype("string").str.strip()
        tokens = set(s.dropna().unique()) & placeholders
        if tokens:
            found[c] = sorted(tokens)
            s = s.replace(list(tokens), np.nan)
        train[c] = s.astype(object).where(s.notna(), np.nan)
print(f"Placeholders found in {len(found)} columns: {sorted(found)}")

drop_cols = set(targets + excluded + ([group_col] if group_col else []))
candidate = [c for c in train.columns if c not in drop_cols]
feat = []
for c in candidate:
    s = train[c]
    if s.dtype.kind in "OUS" and s.nunique(dropna=False) == len(train):
        print(f"Excluded key-like text column: {c}")
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(train):
        vals = np.sort(s.dropna().to_numpy())
        if len(vals) == len(train) and np.all(np.diff(vals) == 1):
            print(f"Excluded key-like integer sequence: {c}")
            continue
    feat.append(c)

derived = set()
if targets:
    num_feat = [c for c in feat if pd.api.types.is_numeric_dtype(train[c]) and (train[c].dropna() >= 0).all()]
    for t in targets:
        if t not in train:
            continue
        y = train[t]
        for c in list(feat):
            if train[c].equals(y):
                derived.add(c)
                print(f"Excluded exact derivation of {t}: {c}")
        for r in (2, 3):
            for combo in itertools.combinations(num_feat, r):
                if all(c not in derived for c in combo) and train[list(combo)].sum(axis=1).equals(y):
                    derived.update(combo)
                    print(f"Excluded row-sum derivation of {t}: {combo}")
feat = [c for c in feat if c not in derived]

if targets:
    for t in targets:
        if t in train and pd.api.types.is_numeric_dtype(train[t]):
            for c in feat:
                if pd.api.types.is_numeric_dtype(train[c]):
                    corr = train[[c, t]].corr().iloc[0, 1]
                    if abs(corr) >= 0.98:
                        print(f"Correlation warning >=0.98: {c} with {t}")

for c in list(feat):
    if pd.api.types.is_datetime64_any_dtype(train[c]):
        train[c + "_month"] = train[c].dt.month
        train[c + "_dow"] = train[c].dt.dayofweek
        train[c + "_hour"] = train[c].dt.hour
        feat.remove(c)
        feat.extend([c + "_month", c + "_dow", c + "_hour"])

for c in list(feat):
    if train[c].dtype.kind in "OUS" and any(k in c.lower() for k in ("date", "time")):
        try:
            parsed = pd.to_datetime(train[c], errors="raise")
            train[c + "_month"] = parsed.dt.month
            train[c + "_dow"] = parsed.dt.dayofweek
            train[c + "_hour"] = parsed.dt.hour
            feat.remove(c)
            feat.extend([c + "_month", c + "_dow", c + "_hour"])
        except Exception:
            pass

numeric_cols = [c for c in feat if pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_bool_dtype(train[c])]
cat_cols = [c for c in feat if c not in numeric_cols]
thr = max(1, int(0.05 * len(train)))
for c in list(cat_cols):
    if train[c].nunique(dropna=False) > thr:
        print(f"Excluded high-cardinality categorical: {c}")
        cat_cols.remove(c)
        feat.remove(c)

n_cat_out = sum(train[c].nunique(dropna=False) for c in cat_cols)
n_out = len(numeric_cols) + n_cat_out
sparse = n_out > 300

num_ts = []
for c in numeric_cols:
    s = train[c]
    steps = []
    if (s.dropna() >= 0).all() and s.dropna().skew() > 1:
        steps.append(("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    steps.append(("impute", SimpleImputer(strategy="median")))
    steps.append(("scale", StandardScaler()))
    num_ts.append((f"num_{c}", Pipeline(steps), [c]))

cat_ts = []
for c in cat_cols:
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)),
    ])
    cat_ts.append((f"cat_{c}", pipe, [c]))

transformers = num_ts + cat_ts
pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
if n_out <= 300:
    pre.set_output(transform="pandas")
    for _, pipe, _ in transformers:
        pipe.set_output(transform="pandas")

X_train = train[feat].copy()
pre = mlca_runtime.fit_record(pre, X_train, "train")
mlca_runtime.save_model(pre, "preprocessor.pkl")
print(f"Fitted preprocessor: {len(feat)} features, {n_out} encoded columns")
