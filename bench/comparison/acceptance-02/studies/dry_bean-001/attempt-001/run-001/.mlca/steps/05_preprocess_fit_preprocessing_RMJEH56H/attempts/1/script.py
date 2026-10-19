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

def need(d, key):
    if key not in d:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {sorted(d.keys())}")
    return d[key]

targets = list(need(protocol, "targets"))
excluded = set(need(protocol, "excluded_columns") or [])
group_col = need(protocol, "group_column")
_ = need(protocol, "task")
for t in targets:
    if t not in train.columns:
        raise KeyError(f"Target column {t!r} not found in train")

base_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]

tokens = {"?", "-", "NA", "N/A"}
ph_cols = []
for c in base_cols:
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]):
        raw = train[c].astype("string")
        stripped = raw.str.strip()
        mask = stripped.isin(tokens) | (raw.notna() & stripped.eq(""))
        if mask.any():
            ph_cols.append(c)
print("Placeholders in object columns: " + (", ".join(ph_cols) if ph_cols else "none"))

date_cols = []
for c in list(base_cols):
    if pd.api.types.is_datetime64_any_dtype(train[c]):
        date_cols.append(c)
    elif (pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])) and any(
        tok in c.lower() for tok in ["date", "time", "_dt"]
    ):
        parsed = pd.to_datetime(train[c], errors="coerce")
        if parsed.notna().mean() > 0.9:
            train[c] = parsed
            val[c] = pd.to_datetime(val[c], errors="coerce")
            date_cols.append(c)
for c in date_cols:
    train[f"{c}_month"] = train[c].dt.month
    train[f"{c}_dow"] = train[c].dt.dayofweek
    train[f"{c}_hour"] = train[c].dt.hour
    train[f"{c}_days"] = (train[c] - pd.Timestamp("2000-01-01")).dt.days
    val[f"{c}_month"] = val[c].dt.month
    val[f"{c}_dow"] = val[c].dt.dayofweek
    val[f"{c}_hour"] = val[c].dt.hour
    val[f"{c}_days"] = (val[c] - pd.Timestamp("2000-01-01")).dt.days
    base_cols.remove(c)
    base_cols += [f"{c}_month", f"{c}_dow", f"{c}_hour", f"{c}_days"]

key_drop = []
for c in list(base_cols):
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=False) == len(train):
            key_drop.append((c, "all-distinct text/code"))
            base_cols.remove(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=False) == len(train) and s.sort_values().diff().dropna().eq(1).all():
            key_drop.append((c, "integer sequence"))
            base_cols.remove(c)
for c in list(base_cols):
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=False) > 0.05 * len(train):
            key_drop.append((c, "high-cardinality text/code"))
            base_cols.remove(c)

deriv = []
for t in targets:
    tgt = train[t]
    for c in list(base_cols):
        if train[c].dtype == tgt.dtype and train[c].equals(tgt):
            deriv.append((c, f"exact target {t}"))
            if c in base_cols:
                base_cols.remove(c)
    if pd.api.types.is_numeric_dtype(tgt):
        num_c = [c for c in base_cols if pd.api.types.is_numeric_dtype(train[c]) and (train[c].dropna() >= 0).all()]
        for k in [1, 2, 3]:
            found = False
            for combo in itertools.combinations(num_c, k):
                s = train[list(combo)].sum(axis=1)
                if np.allclose(s.values, tgt.values, equal_nan=True):
                    for c in combo:
                        if c in base_cols:
                            deriv.append((c, f"sum with {combo} equals target {t}"))
                            base_cols.remove(c)
                    found = True
                    break
            if found:
                break

num_cols, cat_cols = [], []
for c in base_cols:
    s = train[c]
    if (
        pd.api.types.is_bool_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
    ):
        cat_cols.append(c)
    else:
        num_cols.append(c)

est_width = len(num_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
narrow = est_width <= 300

transformers = []
if num_cols:
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not narrow)),
    ])
    transformers.append(("cat", cat_pipe, cat_cols))
if not transformers:
    raise ValueError("No feature columns remain after screening")

pre = ColumnTransformer(transformers, remainder="drop")
if narrow:
    pre.set_output(transform="pandas")
    for _, pipe, _ in transformers:
        pipe.set_output(transform="pandas")

X_train = train[base_cols].copy()
X_val = val[base_cols].copy()
fitted = mlca_runtime.fit_record(pre, X_train, "train")
if fitted is not None:
    pre = fitted
X_val_t = pre.transform(X_val)

print(f"Features: {len(num_cols)} numeric, {len(cat_cols)} categorical; encoded width {X_val_t.shape[1]}")
if key_drop:
    print("Dropped key-like: " + "; ".join(f"{c} ({r})" for c, r in key_drop))
if deriv:
    print("Excluded target derivations: " + "; ".join(f"{c} ({r})" for c, r in deriv))
print(f"Train transform rows {X_train.shape[0]}; val transform rows {X_val_t.shape[0]}")

mlca_runtime.save_model(pre, "preprocessor.pkl")
