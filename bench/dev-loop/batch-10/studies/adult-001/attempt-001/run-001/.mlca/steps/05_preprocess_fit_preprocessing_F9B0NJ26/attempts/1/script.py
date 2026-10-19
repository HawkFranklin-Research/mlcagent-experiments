import json
import itertools
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
if "group_column" not in protocol:
    raise ValueError(f"Missing group_column; available keys: {list(protocol)}")
group_col = protocol["group_column"]

# Define feature columns once
feature_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group_col
]

# Drop key-like columns: text codes or integer sequences all distinct and carrying no measurement
key_like = []
for c in feature_cols:
    s = train[c]
    if s.dtype == "object" or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=True) == len(s):
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=True) == len(s) and any(k in c.lower() for k in ("id", "key", "index", "code")):
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

# Detect and clean placeholder tokens and whitespace
placeholders = {"?", "-", "NA", "N/A"}
placeholder_cols = []
for c in feature_cols:
    s = train[c]
    if s.dtype == "object" or pd.api.types.is_string_dtype(s):
        stripped = s.astype(str).str.strip()
        bad = stripped.isin(placeholders) | (s.astype(str) != stripped)
        if bad.any():
            placeholder_cols.append(c)
if placeholder_cols:
    print(f"Placeholders in {len(placeholder_cols)} columns: {placeholder_cols}")

def clean_text(df):
    df = df.copy()
    for c in feature_cols:
        s = df[c]
        if s.dtype == "object" or pd.api.types.is_string_dtype(s):
            s = s.astype(str).str.strip().replace(list(placeholders), np.nan)
            df[c] = s
    return df

train = clean_text(train)
val = clean_text(val)

# Screen target derivations
if len(targets) == 1:
    y_train = train[targets[0]]
    derivations = []
    for c in feature_cols:
        if train[c].astype(str).equals(y_train.astype(str)):
            derivations.append((c, "exact equality"))
    if pd.api.types.is_numeric_dtype(y_train):
        nonneg = [
            c for c in feature_cols
            if pd.api.types.is_numeric_dtype(train[c]) and train[c].min() >= 0
        ]
        for r in (2, 3):
            for combo in itertools.combinations(nonneg, r):
                if any(c in [d[0] for d in derivations] for c in combo):
                    continue
                if np.allclose(train[list(combo)].sum(axis=1).values, y_train.values):
                    for c in combo:
                        derivations.append((c, f"sum of {combo}"))
                    break
    for c, reason in derivations:
        print(f"Excluding feature {c}: {reason}")
    excluded_derived = {d[0] for d in derivations}
    feature_cols = [c for c in feature_cols if c not in excluded_derived]

# Sort feature columns into numeric and categorical
num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]

# Handle datetime columns: derive numeric parts, drop raw
date_cols = [c for c in feature_cols if pd.api.types.is_datetime64_any_dtype(train[c])]
for c in date_cols:
    for part, fn in (
        ("year", lambda s: s.dt.year),
        ("month", lambda s: s.dt.month),
        ("day", lambda s: s.dt.day),
        ("dayofweek", lambda s: s.dt.dayofweek),
    ):
        nc = f"{c}_{part}"
        train[nc] = fn(train[c])
        val[nc] = fn(val[c])
        num_cols.append(nc)
    feature_cols.remove(c)
    train = train.drop(columns=[c])
    val = val.drop(columns=[c])
    cat_cols = [x for x in cat_cols if x != c]

# Drop high-cardinality categorical columns (>5% unique)
high_card = [c for c in cat_cols if train[c].nunique(dropna=True) > 0.05 * len(train)]
if high_card:
    print(f"Dropping high-cardinality categorical columns: {high_card}")
    cat_cols = [c for c in cat_cols if c not in high_card]
    feature_cols = [c for c in feature_cols if c not in high_card]

final_features = num_cols + cat_cols
X_train = train[final_features]
X_val = val[final_features]

# Build preprocessing: numeric median impute + standard scale; categorical most-frequent impute + one-hot
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ],
    remainder="drop",
)

approx_width = len(num_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
if approx_width <= 300:
    preprocessor.set_output(transform="pandas")

preprocessor = mlca_runtime.fit_record(
    preprocessor, X_train, "train", y=train[targets[0]]
)
Xt_val = preprocessor.transform(X_val)
print(f"Fitted preprocessor; train shape {X_train.shape}, val encoded shape {Xt_val.shape}")

mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
