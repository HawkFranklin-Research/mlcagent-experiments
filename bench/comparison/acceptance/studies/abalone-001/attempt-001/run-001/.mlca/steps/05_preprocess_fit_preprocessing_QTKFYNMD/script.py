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

for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available: {list(protocol)}")

targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")
if len(targets) != 1:
    raise ValueError(f"Expected exactly one target, got {targets}")
target = targets[0]

train = pd.read_parquet(inputs["train"])
n = len(train)

def is_key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=False) == n
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.nunique(dropna=False) == n and (s.is_monotonic_increasing or s.is_monotonic_decreasing)
    return False

features = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group and not is_key_like(train[c])
]

tokens = ["?", "-", "NA", "N/A", ""]
placeholder_cols = []
for c in features:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        st = s.astype("string").str.strip()
        if st.isin(tokens).any():
            placeholder_cols.append(c)
print(f"placeholders in {len(placeholder_cols)} columns: {placeholder_cols}")

y = train[target]
excluded_reasons = {}
for c in features:
    if np.array_equal(train[c].to_numpy(), y.to_numpy()):
        excluded_reasons[c] = f"equals {target}"

if pd.api.types.is_numeric_dtype(y):
    num_c = [
        c for c in features
        if c not in excluded_reasons
        and pd.api.types.is_numeric_dtype(train[c])
        and bool((train[c] >= 0).all())
    ]
    for r in (2, 3):
        for combo in combinations(num_c, r):
            if any(c in excluded_reasons for c in combo):
                continue
            if np.array_equal(train[list(combo)].sum(axis=1).to_numpy(), y.to_numpy()):
                for c in combo:
                    excluded_reasons[c] = f"sum of {combo} equals {target}"

features = [c for c in features if c not in excluded_reasons]
print("excluded: " + ("; ".join(f"{c} ({r})" for c, r in excluded_reasons.items()) if excluded_reasons else "none"))

if pd.api.types.is_numeric_dtype(y):
    warns = []
    for c in features:
        if pd.api.types.is_numeric_dtype(train[c]):
            corr = train[[c, target]].corr().iloc[0, 1]
            if pd.notna(corr) and abs(corr) >= 0.98:
                warns.append(c)
    if warns:
        print(f"corr>=0.98 kept: {warns}")

num_cols, cat_cols = [], []
for c in features:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        st = s.astype("string").str.strip()
        if st.nunique(dropna=True) > max(1, int(0.05 * n)):
            continue
        cat_cols.append(c)

features = num_cols + cat_cols
if not features:
    raise ValueError("No feature columns remain after protocol exclusions and screening")

width = len(num_cols)
for c in cat_cols:
    width += train[c].astype("string").str.strip().nunique(dropna=True) + 1
sparse = width > 300

transformers = []
if num_cols:
    transformers.append((
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]),
        num_cols,
    ))
if cat_cols:
    transformers.append((
        "cat",
        Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)),
        ]),
        cat_cols,
    ))

pre = ColumnTransformer(transformers, remainder="drop")
if not sparse:
    pre.set_output(transform="pandas")

X = train[features]
mlca_runtime.fit_record(pre, X, "train")
mlca_runtime.save_model(pre, "preprocessor.pkl")
print(f"fitted preprocessor on {X.shape[1]} features; estimated width {width}")
