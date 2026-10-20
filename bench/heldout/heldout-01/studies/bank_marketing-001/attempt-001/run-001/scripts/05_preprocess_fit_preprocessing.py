import json
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

try:
    targets = protocol["targets"]
    excluded = set(protocol["excluded_columns"])
    task = protocol["task"]
except KeyError as e:
    raise KeyError(
        f"Missing protocol key {e}; available top-level keys: {list(protocol)}"
    )
group_col = protocol.get("group_column")

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])
target = targets[0]
y = train[target]

drop_cols = set(targets) | excluded | ({group_col} if group_col else set())
candidates = [c for c in train.columns if c not in drop_cols]

feature_cols = []
for c in candidates:
    s = train[c]
    is_textish = (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )
    if is_textish and s.nunique(dropna=True) == len(s) and len(s) > 20:
        continue
    feature_cols.append(c)

ph_tokens = {"?", "-", "NA", "N/A", ""}
ph_cols, ph_found = [], set()
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        vals = set(s.dropna().astype(str).str.strip().unique())
        hit = vals & ph_tokens
        if hit:
            ph_cols.append(c)
            ph_found |= hit
print("placeholders: " + ("none (0 columns)" if not ph_cols else f"{sorted(ph_found)} in {len(ph_cols)} columns"))

def find_derivations(df, cols, target_col):
    out = {}
    yv = df[target_col]
    for c in cols:
        try:
            if (df[c].astype(object) == yv.astype(object)).all():
                out[c] = "equals target"
        except Exception:
            pass
    if pd.api.types.is_numeric_dtype(yv):
        num = [
            c for c in cols
            if pd.api.types.is_numeric_dtype(df[c])
            and not df[c].isna().any()
            and (df[c] >= 0).all()
        ]
        for r in range(1, 4):
            for combo in combinations(num, r):
                s = df[list(combo)].sum(axis=1)
                if (s == yv).all():
                    for c in combo:
                        out.setdefault(c, f"sum {combo} equals target")
    return out

derivations = find_derivations(train, feature_cols, target)
feature_cols = [c for c in feature_cols if c not in derivations]
print("excluded derivations: " + ("none" if not derivations else ", ".join(f"{k} ({v})" for k, v in derivations.items())))

warn = []
if pd.api.types.is_numeric_dtype(y) and y.nunique() > 2:
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train[c]):
            corr = train[c].corr(y)
            if abs(corr) >= 0.98:
                warn.append(c)
print("high-correlation warnings: " + ("none" if not warn else ", ".join(warn)))

n = len(train)
num_cols, cat_cols = [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    else:
        if s.nunique(dropna=True) > 0.05 * n:
            continue
        cat_cols.append(c)

width = len(num_cols) + sum(max(1, train[c].nunique(dropna=True)) for c in cat_cols)
small = width <= 300

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not small)),
])

transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))

ct = ColumnTransformer(transformers, remainder="drop")
if small:
    ct.set_output(transform="pandas")

steps = []
if ph_cols:
    steps.extend([
        ("strip", FunctionTransformer(
            pd.DataFrame.replace,
            kw_args={"to_replace": r"^\s+|\s+$", "value": "", "regex": True},
            validate=False,
        )),
        ("placeholders_to_nan", FunctionTransformer(
            pd.DataFrame.replace,
            kw_args={"to_replace": r"^(?:\?|-|NA|N/A)?$", "value": np.nan, "regex": True},
            validate=False,
        )),
    ])
steps.append(("ct", ct))
preprocessor = Pipeline(steps) if len(steps) > 1 else ct

preprocessor = mlca_runtime.fit_record(preprocessor, train[feature_cols], "train")
val_transformed = preprocessor.transform(val[feature_cols])
print(f"preprocessor fitted on train rows={len(train)} features={len(feature_cols)}; val transformed shape={val_transformed.shape}")

mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
