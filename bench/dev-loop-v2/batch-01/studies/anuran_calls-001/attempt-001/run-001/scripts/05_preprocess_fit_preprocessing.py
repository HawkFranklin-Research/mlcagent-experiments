import json
import numpy as np
import pandas as pd
import mlca_runtime
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise ValueError(f"Missing protocol keys {missing}; available: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol["group_column"]
train = pd.read_parquet(paths["train"])
if any(t not in train.columns for t in targets):
    raise ValueError("A protocol target is missing from the training table")

def is_text(s):
    return s.dtype == object or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype)

excluded_set = set(targets) | set(excluded) | {group_col}
feature_cols = []
for c in train.columns:
    if c in excluded_set:
        continue
    s = train[c]
    if is_text(s) and s.notna().all() and s.nunique(dropna=False) == len(train):
        continue
    if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=False) == len(train):
        if any(k in str(c).lower() for k in ("id", "code", "record", "key")):
            continue
    feature_cols.append(c)

derived = {}
numeric_candidates = [
    c for c in feature_cols
    if pd.api.types.is_numeric_dtype(train[c].dtype)
    and not pd.api.types.is_bool_dtype(train[c].dtype)
    and train[c].notna().all()
    and (train[c] >= 0).all()
]
for t in targets:
    y = train[t]
    for c in feature_cols:
        if c in derived:
            continue
        if train[c].notna().all() and y.notna().all():
            try:
                if bool((train[c] == y).all()):
                    derived[c] = f"exact copy of {t}"
            except Exception:
                pass
    if pd.api.types.is_numeric_dtype(y.dtype) and not pd.api.types.is_bool_dtype(y.dtype):
        cand = [c for c in numeric_candidates if c not in derived]
        for r in (1, 2, 3):
            for combo in combinations(cand, r):
                if any(c in derived for c in combo):
                    continue
                mat = train[list(combo)]
                if (mat < 0).any().any() or mat.isna().any().any():
                    continue
                if np.allclose(mat.sum(axis=1).to_numpy(), y.to_numpy(), rtol=0, atol=0, equal_nan=False):
                    for c in combo:
                        derived.setdefault(c, f"row sum to {t}")

corr_warn = []
for t in targets:
    y = train[t]
    if pd.api.types.is_numeric_dtype(y.dtype) and not pd.api.types.is_bool_dtype(y.dtype):
        for c in feature_cols:
            if c in derived:
                continue
            s = train[c]
            if pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype):
                try:
                    r = abs(float(np.corrcoef(s.astype(float), y.astype(float))[0, 1]))
                    if r >= 0.98:
                        corr_warn.append(c)
                except Exception:
                    pass

final_features = [c for c in feature_cols if c not in derived]
numeric_features, categorical_features, high_card, date_features = [], [], [], []
for c in final_features:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        date_features.append(c)
    elif pd.api.types.is_numeric_dtype(s.dtype) or pd.api.types.is_bool_dtype(s.dtype):
        numeric_features.append(c)
    elif is_text(s):
        n = s.nunique(dropna=True)
        if n <= max(1, int(0.05 * len(train))):
            categorical_features.append(c)
        else:
            high_card.append(c)
    else:
        high_card.append(c)

if date_features:
    raise ValueError("Date features detected; built-in-only preprocessor cannot parse them")

final_features = [c for c in final_features if c not in high_card]
placeholders = {"?", "-", "NA", "N/A", ""}
placeholder_cols = []
for c in final_features:
    s = train[c]
    if is_text(s):
        vals = s.dropna().astype(str)
        if (vals.str.strip().isin(placeholders) | vals.str.fullmatch(r"\s*")).any():
            placeholder_cols.append(c)
if placeholder_cols:
    raise ValueError("Placeholder tokens in text columns require a cleaning transformer")

cat_width = sum(max(1, int(train[c].nunique(dropna=True))) for c in categorical_features)
use_pandas = len(numeric_features) + cat_width <= 300
transformers = []
if numeric_features:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    if use_pandas:
        num_pipe.set_output(transform="pandas")
    transformers.append(("num", num_pipe, numeric_features))
if categorical_features:
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not use_pandas)),
    ])
    if use_pandas:
        cat_pipe.set_output(transform="pandas")
    transformers.append(("cat", cat_pipe, categorical_features))
if not transformers:
    raise ValueError("No usable feature columns after screening")

preprocessor = ColumnTransformer(transformers)
if use_pandas:
    preprocessor.set_output(transform="pandas")
preprocessor = mlca_runtime.fit_record(preprocessor, train[final_features], partition="train")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

lines = [
    f"Train shape {train.shape}; used {len(final_features)} features "
    f"(numeric {len(numeric_features)}, categorical {len(categorical_features)}, "
    f"dropped_high_card {len(high_card)}, derived_excluded {len(derived)})"
]
if derived:
    lines.append("Derived exclusions: " + ", ".join(f"{c}: {r}" for c, r in list(derived.items())[:10]) + (" ..." if len(derived) > 10 else ""))
extra = []
if high_card:
    extra.append("Dropped high-card " + ", ".join(high_card[:10]))
if corr_warn:
    extra.append("Corr>=0.98 kept " + ", ".join(sorted(set(corr_warn))[:10]))
if extra:
    lines.append("; ".join(extra))
lines.append("Saved preprocessor.pkl")
print("\n".join(lines[:5]))
