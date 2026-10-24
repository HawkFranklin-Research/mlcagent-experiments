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
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
protocol = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise ValueError(f"Missing protocol key {key}; available keys: {list(protocol)}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

work_train, work_val = train.copy(), val.copy()
exclude_set = set(targets) | set(excluded)
if group_col:
    exclude_set.add(group_col)

base_candidates = [c for c in work_train.columns if c not in exclude_set]
date_cols = []
for c in base_candidates:
    s = work_train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        try:
            parsed = pd.to_datetime(s.dropna().head(200), errors="coerce")
            if len(parsed) and parsed.notna().mean() >= 0.9:
                date_cols.append(c)
        except Exception:
            pass
for c in date_cols:
    dt = pd.to_datetime(work_train[c], errors="coerce")
    dtv = pd.to_datetime(work_val[c], errors="coerce")
    for part in ["month", "dayofweek", "hour"]:
        work_train[f"{c}_{part}"] = getattr(dt.dt, part)
        work_val[f"{c}_{part}"] = getattr(dtv.dt, part)

base_candidates = [c for c in work_train.columns if c not in exclude_set and c not in date_cols]
n_rows = len(work_train)
key_like = []
for c in base_candidates:
    s = work_train[c]
    if s.nunique(dropna=True) == n_rows:
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s):
            key_like.append(c)
base_candidates = [c for c in base_candidates if c not in key_like]

high_card = []
for c in base_candidates:
    s = work_train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) > 0.05 * n_rows:
        high_card.append(c)
feature_cols = [c for c in base_candidates if c not in high_card]

derived = set()
for target in targets:
    if target not in work_train:
        continue
    y = work_train[target]
    for c in feature_cols:
        s = work_train[c]
        if s.dtype == y.dtype and ((s == y) | (s.isna() & y.isna())).all():
            derived.add(c)
    if pd.api.types.is_numeric_dtype(y) and not y.isna().any():
        nonneg = [c for c in feature_cols if c not in derived and pd.api.types.is_numeric_dtype(work_train[c])
                  and work_train[c].notna().all() and (work_train[c] >= 0).all()]
        for k in [1, 2, 3]:
            for combo in itertools.combinations(nonneg, k):
                if (work_train[list(combo)].sum(axis=1) == y).all():
                    derived.update(combo)
                    break
            if derived:
                break
if derived:
    print("Excluded target-derived features:", ", ".join(sorted(derived)))
feature_cols = [c for c in feature_cols if c not in derived]

corr_warn = []
for target in targets:
    if target in work_train and pd.api.types.is_numeric_dtype(work_train[target]):
        for c in feature_cols:
            if pd.api.types.is_numeric_dtype(work_train[c]):
                corr = work_train[[c, target]].corr().iloc[0, 1]
                if pd.notna(corr) and abs(corr) >= 0.98:
                    corr_warn.append(c)
if corr_warn:
    print("High correlation warning (kept):", ", ".join(sorted(set(corr_warn))))

placeholder_cols = []
for c in feature_cols:
    s = work_train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.astype(str).str.match(r"^\s*(?:\?|-|NA|N/A)?\s*$").any():
            placeholder_cols.append(c)
print(f"Placeholder/missing-like tokens found in {len(placeholder_cols)} text columns")

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(work_train[c])]
cat_cols = [c for c in feature_cols if c not in numeric_cols]
cat_width = sum(max(1, work_train[c].nunique(dropna=True)) for c in cat_cols)
estimated_width = len(numeric_cols) + cat_width
pandas_out = estimated_width <= 300
sparse_out = not pandas_out

strip = FunctionTransformer(pd.DataFrame.replace, kw_args={"to_replace": r"^\s+|\s+$", "value": "", "regex": True})
blank = FunctionTransformer(pd.DataFrame.replace, kw_args={"to_replace": r"^\s*(?:\?|-|NA|N/A)?\s*$", "value": np.nan, "regex": True})
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_out))])
transformers = []
if numeric_cols:
    transformers.append(("num", num_pipe, numeric_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
preprocessor = Pipeline([("strip", strip), ("blank", blank), ("pre", ct)])
if pandas_out:
    ct.set_output(transform="pandas")
    preprocessor.set_output(transform="pandas")

X_train = work_train[feature_cols].copy()
X_val = work_val[feature_cols].copy()
fitted = mlca_runtime.fit_record(preprocessor, X_train, "train")
if fitted is not None:
    preprocessor = fitted
Xt_val = preprocessor.transform(X_val)
encoded_width = Xt_val.shape[1] if hasattr(Xt_val, "shape") else len(preprocessor.get_feature_names_out())
mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_encoded_features", float(encoded_width), "train")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
print(f"train/val rows: {len(work_train)}/{len(work_val)}; features: {len(feature_cols)}")
print(f"numeric/categorical: {len(numeric_cols)}/{len(cat_cols)}; encoded width: {encoded_width}")
print("Saved preprocessor.pkl")
