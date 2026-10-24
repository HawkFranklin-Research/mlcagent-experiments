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
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
protocol = json.loads(inputs["_protocol"].read_text())

targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")
y = train[targets[0]] if targets else None
n = len(train)

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]

key_like = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        if s.nunique(dropna=True) == n:
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s.dtype):
        if s.nunique(dropna=True) == n and any(t in str(c).lower() for t in ("id", "key", "code")):
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

derived = []
if y is not None:
    for c in feature_cols:
        s = train[c]
        if s.dtype == y.dtype and s.equals(y):
            derived.append((c, "exact target copy"))
        elif pd.api.types.is_numeric_dtype(s.dtype) and pd.api.types.is_numeric_dtype(y.dtype):
            if np.allclose(s.to_numpy(dtype=float, na_value=np.nan),
                           y.to_numpy(dtype=float, na_value=np.nan),
                           equal_nan=True):
                derived.append((c, "exact target copy"))

    if pd.api.types.is_numeric_dtype(y.dtype) and set(pd.unique(y.dropna())).issubset({0, 1}):
        ybin = y.fillna(0).astype(int).to_numpy()
        candidates = []
        for c in feature_cols:
            s = train[c]
            if pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_datetime64_any_dtype(s.dtype):
                arr = s.to_numpy(dtype=float, na_value=np.nan)
                if np.all(np.isfinite(arr)) and np.nanmin(arr) >= 0 and np.all(arr[ybin == 0] == 0):
                    candidates.append(c)
        if 0 < len(candidates) <= 100:
            for k in (1, 2, 3):
                for combo in combinations(candidates, k):
                    ssum = np.sum(train[list(combo)].to_numpy(dtype=float), axis=1)
                    if np.allclose(ssum, ybin.astype(float), equal_nan=False):
                        for c in combo:
                            if c not in {d[0] for d in derived}:
                                derived.append((c, "row-sum derivation"))
                        break

derived_names = {c for c, _ in derived}
selected_features = [c for c in feature_cols if c not in derived_names]

corr_warn = []
if y is not None and pd.api.types.is_numeric_dtype(y.dtype):
    yv = y.to_numpy(dtype=float, na_value=np.nan)
    for c in selected_features:
        s = train[c]
        if pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_datetime64_any_dtype(s.dtype):
            xv = s.to_numpy(dtype=float, na_value=np.nan)
            ok = np.isfinite(xv).all() and np.isfinite(yv).all()
            if ok and np.std(xv) > 0 and np.std(yv) > 0:
                if abs(np.corrcoef(xv, yv)[0, 1]) >= 0.98:
                    corr_warn.append(c)

text_cols = [c for c in selected_features
             if pd.api.types.is_object_dtype(train[c].dtype) or pd.api.types.is_string_dtype(train[c].dtype)]
placeholder_cols = []
placeholders = {"?", "-", "na", "n/a", ""}
for c in text_cols:
    s = train[c].dropna().astype(str).str.strip().str.lower()
    if s.isin(placeholders).any():
        placeholder_cols.append(c)

numeric_cols, categorical_cols, date_cols = [], [], []
for c in selected_features:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s.dtype):
        numeric_cols.append(c)
    else:
        if s.nunique(dropna=True) <= max(1, int(0.05 * n)):
            categorical_cols.append(c)

final_features = numeric_cols + categorical_cols
onehot_width = sum(train[c].nunique(dropna=True) for c in categorical_cols)
total_width = len(numeric_cols) + onehot_width
wide = total_width > 300

numeric_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=wide)),
])
transformers = [("num", numeric_tf, numeric_cols)]
if categorical_cols:
    transformers.append(("cat", categorical_tf, categorical_cols))

preprocessor = ColumnTransformer(transformers, remainder="drop")
if not wide:
    preprocessor.set_output(transform="pandas")

preprocessor = mlca_runtime.fit_record(preprocessor, train[final_features], "train")
val_X = preprocessor.transform(val[final_features])
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
mlca_runtime.record_metric("n_features", float(len(final_features)), "train")

print(f"features={len(final_features)} numeric={len(numeric_cols)} categorical={len(categorical_cols)} wide={wide}")
print(f"key_like_dropped={len(key_like)} derived_excluded={len(derived_names)} corr_ge_098={len(corr_warn)}")
print(f"placeholder_text_cols={len(placeholder_cols)}")
print(f"val_transformed_shape={val_X.shape}")
