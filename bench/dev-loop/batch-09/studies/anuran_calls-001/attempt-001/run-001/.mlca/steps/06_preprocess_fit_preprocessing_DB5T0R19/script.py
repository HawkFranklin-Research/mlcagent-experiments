import json
import itertools
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available keys: {list(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]
train = pd.read_parquet(inputs["train"])

for c in targets + [group]:
    if c not in train.columns:
        raise KeyError(f"Required protocol column {c!r} not found in train; available: {list(train.columns)}")

drop = set(targets + excluded + [group])
candidate = [c for c in train.columns if c not in drop]

key_like = []
for c in candidate:
    s = train[c]
    if (s.dtype == object or pd.api.types.is_string_dtype(s.dtype)) and s.nunique(dropna=False) == len(train) > 1:
        key_like.append(c)
    elif pd.api.types.is_integer_dtype(s.dtype) and not set(s.dropna().unique()).issubset({0, 1}):
        if s.nunique(dropna=False) == len(train) > 1:
            vals = np.sort(s.dropna().unique())
            if len(vals) and np.array_equal(vals, np.arange(vals.min(), vals.max() + 1)):
                key_like.append(c)
candidate = [c for c in candidate if c not in key_like]

excluded_deriv = set()
for t in targets:
    y = train[t]
    for c in candidate:
        if c not in excluded_deriv and train[c].equals(y):
            excluded_deriv.add(c)
            print(f"Excluded {c}: exact copy of {t}")
    if pd.api.types.is_numeric_dtype(y):
        num_cands = [c for c in candidate if pd.api.types.is_numeric_dtype(train[c]) and (train[c] >= 0).all()]
        for r in (1, 2, 3):
            for combo in itertools.combinations(num_cands, r):
                if any(c in excluded_deriv for c in combo):
                    continue
                summed = train[list(combo)].sum(axis=1)
                if np.allclose(summed.to_numpy(), y.to_numpy(), rtol=0, atol=0, equal_nan=True):
                    for c in combo:
                        if c not in excluded_deriv:
                            excluded_deriv.add(c)
                            print(f"Excluded {c}: sum derivation of {t}")

warn = []
for t in targets:
    y = train[t]
    if pd.api.types.is_numeric_dtype(y):
        for c in candidate:
            if c in excluded_deriv or not pd.api.types.is_numeric_dtype(train[c]):
                continue
            corr = train[[c, t]].corr().iloc[0, 1]
            if pd.notna(corr) and abs(corr) >= 0.98:
                warn.append(c)
if warn:
    print("Correlation>=0.98 kept: " + ", ".join(sorted(set(warn))))

features = [c for c in candidate if c not in excluded_deriv]
text_cols = [c for c in features if train[c].dtype == object or pd.api.types.is_string_dtype(train[c].dtype)]
placeholder_count = 0
for c in text_cols:
    s = train[c].dropna().astype(str)
    stripped = s.str.strip()
    mask = s.isin(["?", "-", "NA", "N/A"]) | (s != stripped) | (stripped == "")
    if mask.any():
        placeholder_count += 1
print(f"Text placeholder columns: {placeholder_count}")

num_cols, cat_cols, date_cols, high_card = [], [], [], []
threshold = max(1, int(0.05 * len(train)))
for c in features:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    elif s.dtype == object or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=False) <= threshold:
            cat_cols.append(c)
        else:
            high_card.append(c)
    elif s.nunique(dropna=False) <= threshold:
        cat_cols.append(c)
    else:
        high_card.append(c)

if date_cols:
    print("Dropped datetime columns: " + ", ".join(date_cols))
features = num_cols + cat_cols
est_width = len(num_cols) + sum(train[c].nunique(dropna=False) + 1 for c in cat_cols)
use_pandas = est_width <= 300

strip_ws = FunctionTransformer(pd.DataFrame.replace, kw_args={"to_replace": r"^\s+|\s+$", "value": "", "regex": True})
to_missing = FunctionTransformer(
    pd.DataFrame.replace,
    kw_args={"to_replace": [r"^\?$", r"^-$", r"^NA$", r"^N/A$", r"^$"], "value": np.nan, "regex": True},
)
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([
    ("strip", strip_ws),
    ("placeholder", to_missing),
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not use_pandas)),
])

transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
preprocessor = ColumnTransformer(transformers, remainder="drop")

if use_pandas:
    preprocessor.set_output(transform="pandas")
    num_pipe.set_output(transform="pandas")
    cat_pipe.set_output(transform="pandas")

X_train = train[features].copy()
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, "train")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
print(f"Saved preprocessor: {len(features)} features ({len(num_cols)} numeric, {len(cat_cols)} categorical)")
