import mlca_runtime
import json, itertools
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for k in ["targets", "excluded_columns", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing {k}; available top-level keys: {list(protocol.keys())}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column", None)

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
target_col = targets[0]
y_train = train[target_col]

candidate_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]

feature_cols = []
for c in candidate_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=False) == len(train):
            continue
    if pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=False) == len(train):
            continue
    feature_cols.append(c)

excluded_info = []
for c in list(feature_cols):
    if (train[c] == y_train).all():
        excluded_info.append((c, "exact equality with target"))
        feature_cols.remove(c)

nonneg = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) and train[c].dropna().min() >= 0]
max_t = y_train.max()
min_t = y_train.min()
for k in (1, 2, 3):
    for combo in itertools.combinations(nonneg, k):
        if any(c not in feature_cols for c in combo):
            continue
        if sum(train[c].max() for c in combo) < max_t:
            continue
        if sum(train[c].min() for c in combo) > min_t:
            continue
        s = train[list(combo)].sum(axis=1)
        if (s == y_train).all():
            for c in combo:
                if c in feature_cols:
                    feature_cols.remove(c)
            excluded_info.append((combo, "sum equals target"))
            nonneg = [c for c in nonneg if c in feature_cols]

if excluded_info:
    print("Excluded derivations: " + "; ".join(f"{c} ({r})" for c, r in excluded_info))
else:
    print("No exact target derivations found")

placeholder_cols = []
for c in feature_cols:
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]):
        s = train[c].astype(str).str.strip()
        if s.isin(["?", "-", "NA", "N/A", ""]).any():
            placeholder_cols.append(c)
if placeholder_cols:
    print(f"Placeholder tokens found in {len(placeholder_cols)} columns")
    for c in placeholder_cols:
        train[c] = train[c].astype(str).str.strip().replace({"?": np.nan, "-": np.nan, "NA": np.nan, "N/A": np.nan, "": np.nan})
        val[c] = val[c].astype(str).str.strip().replace({"?": np.nan, "-": np.nan, "NA": np.nan, "N/A": np.nan, "": np.nan})
else:
    print("No placeholder tokens found")

for c in list(feature_cols):
    if pd.api.types.is_datetime64_any_dtype(train[c]):
        for df in (train, val):
            df[c + "_dayofweek"] = df[c].dt.dayofweek
            df[c + "_month"] = df[c].dt.month
            df[c + "_hour"] = df[c].dt.hour
            df[c + "_days"] = (df[c] - pd.Timestamp("2000-01-01")).dt.days
        feature_cols.remove(c)
        feature_cols.extend([c + "_dayofweek", c + "_month", c + "_hour", c + "_days"])

numeric_cols = []
categorical_cols = []
for c in list(feature_cols):
    if pd.api.types.is_numeric_dtype(train[c]):
        numeric_cols.append(c)
    else:
        if train[c].nunique(dropna=False) > 0.05 * len(train):
            feature_cols.remove(c)
        else:
            categorical_cols.append(c)

num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
pre = ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)], remainder="drop")
est_width = len(numeric_cols) + sum(train[c].nunique(dropna=False) for c in categorical_cols)
if est_width <= 300:
    num_pipe.set_output(transform="pandas")
    cat_pipe.set_output(transform="pandas")
    pre.set_output(transform="pandas")

pre = mlca_runtime.fit_record(pre, train[feature_cols], "train", y=y_train)
X_val = pre.transform(val[feature_cols])
print(f"Features: {len(feature_cols)} numeric={len(numeric_cols)} categorical={len(categorical_cols)}; val encoded shape={getattr(X_val, 'shape', None)}")

mlca_runtime.save_model(pre, "preprocessor.pkl")
