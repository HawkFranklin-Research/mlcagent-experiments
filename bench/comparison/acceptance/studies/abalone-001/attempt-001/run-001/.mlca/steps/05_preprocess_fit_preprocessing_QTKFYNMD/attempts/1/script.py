import mlca_runtime
import json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(protocol)}")

targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target = targets[0]

train = pd.read_parquet(inputs["train"])
n = len(train)

def is_key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=False) == n
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.nunique(dropna=False) == n and (s.is_monotonic_increasing or s.is_monotonic_decreasing)
    return False

features = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    if is_key_like(train[c]):
        continue
    features.append(c)

tokens = ["?", "-", "NA", "N/A", ""]
placeholder_cols = []
for c in features:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        st = s.astype("string").str.strip()
        if st.isin(tokens).any():
            placeholder_cols.append(c)
print(f"placeholder tokens found in {len(placeholder_cols)} columns: {placeholder_cols}")

y = train[target]
excluded_reasons = {}
for c in features:
    if np.array_equal(train[c].to_numpy(), y.to_numpy()):
        excluded_reasons[c] = f"equals {target}"

if pd.api.types.is_numeric_dtype(y):
    num_candidates = [
        c for c in features
        if c not in excluded_reasons
        and pd.api.types.is_numeric_dtype(train[c])
        and bool((train[c] >= 0).all())
    ]
    for r in (2, 3):
        for combo in combinations(num_candidates, r):
            if any(c in excluded_reasons for c in combo):
                continue
            summed = train[list(combo)].sum(axis=1)
            if np.array_equal(summed.to_numpy(), y.to_numpy()):
                for c in combo:
                    excluded_reasons[c] = f"part of sum {combo} equals {target}"

features = [c for c in features if c not in excluded_reasons]
if excluded_reasons:
    print("excluded derivations: " + "; ".join(f"{c} ({r})" for c, r in excluded_reasons.items()))
else:
    print("excluded derivations: none")

if pd.api.types.is_numeric_dtype(y):
    warnings = []
    for c in features:
        if pd.api.types.is_numeric_dtype(train[c]):
            corr = train[[c, target]].corr().iloc[0, 1]
            if pd.notna(corr) and abs(corr) >= 0.98:
                warnings.append(c)
    if warnings:
        print(f"correlation warnings >=0.98: {warnings}")

num_cols, cat_cols, high_card_cols = [], [], []
for c in features:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    elif pd.api.types.is_datetime64_any_dtype(s):
        high_card_cols.append(c)
    else:
        st = s.astype("string").str.strip().mask(s.astype("string").str.strip().isin(tokens))
        if st.nunique(dropna=True) > max(1, int(0.05 * n)):
            high_card_cols.append(c)
        else:
            cat_cols.append(c)

features = num_cols + cat_cols
width = len(num_cols)
for c in cat_cols:
    st = train[c].astype("string").str.strip().mask(train[c].astype("string").str.strip().isin(tokens))
    width += st.nunique(dropna=True) + (1 if st.isna().any() else 0)
sparse = width > 300

transformers = []
if num_cols:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    if not sparse:
        num_pipe.set_output(transform="pandas")
    transformers.append(("num", num_pipe, num_cols))

if cat_cols:
    steps = [
        ("fill_nan", SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="__MISSING__")),
        ("strip", FunctionTransformer(np.char.strip, validate=False)),
    ]
    for i, tok in enumerate(tokens):
        steps.append((f"repl_{i}", SimpleImputer(missing_values=tok, strategy="constant", fill_value="__MISSING__")))
    steps.append(("restore_missing", SimpleImputer(missing_values="__MISSING__", strategy="constant", fill_value=np.nan)))
    steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)))
    cat_pipe = Pipeline(steps)
    if not sparse:
        cat_pipe.set_output(transform="pandas")
    transformers.append(("cat", cat_pipe, cat_cols))

pre = ColumnTransformer(transformers, remainder="drop")
if not sparse:
    pre.set_output(transform="pandas")

X = train[features]
pre = mlca_runtime.fit_record(pre, X, "train", y=y)
mlca_runtime.save_model(pre, "preprocessor.pkl")
print(f"fitted preprocessor on {X.shape[1]} features; output width {width}")
