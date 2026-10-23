import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths["_protocol"].read_text())
for k in ["targets", "excluded_columns", "task"]:
    if k not in proto:
        raise KeyError(f"missing {k}; available top-level keys: {list(proto)}")

targets = proto["targets"]
excluded = set(proto["excluded_columns"])
group_col = proto.get("group_column")
train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])
target = targets[0]

base = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]

def key_like(s):
    if s.isna().any():
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=True) != len(s):
            return False
        ss = s.sort_values()
        return bool((ss.diff().dropna() == 1).all())
    return False

key_cols = [c for c in base if key_like(train[c])]
features = [c for c in base if c not in key_cols]

def screen_derivations(df, feats, tgt):
    y = df[tgt]
    y_num = y.to_numpy(dtype=float) if pd.api.types.is_numeric_dtype(y) else None
    bad = {}
    for c in feats:
        s = df[c]
        if y_num is not None and pd.api.types.is_numeric_dtype(s):
            if np.allclose(s.to_numpy(dtype=float), y_num, equal_nan=True, atol=1e-12, rtol=0):
                bad[c] = "exact target equality"
        elif s.dtype == y.dtype and s.equals(y):
            bad[c] = "exact target equality"
    if y_num is not None and not np.isnan(y_num).any():
        y_max = np.nanmax(y_num)
        cand = []
        for c in feats:
            if c in bad:
                continue
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                arr = s.to_numpy(dtype=float)
                if not np.isnan(arr).any() and (arr >= -1e-12).all() and (arr <= y_max + 1e-9).all():
                    cand.append(c)
        m = len(cand)
        if m:
            arrs = [df[c].to_numpy(dtype=float) for c in cand]
            sig = {}
            for i, a in enumerate(arrs):
                sig.setdefault(tuple(np.round(a, 10)), []).append(i)
            for i in range(m):
                key = tuple(np.round(y_num - arrs[i], 10))
                for j in sig.get(key, []):
                    if j > i:
                        for idx in (i, j):
                            bad[cand[idx]] = "target = sum of up to 3 non-negative features"
            if m <= 200:
                for i in range(m):
                    base_needed = y_num - arrs[i]
                    for j in range(i + 1, m):
                        key = tuple(np.round(base_needed - arrs[j], 10))
                        for k in sig.get(key, []):
                            if k > j:
                                for idx in (i, j, k):
                                    bad[cand[idx]] = "target = sum of up to 3 non-negative features"
    return bad

bad = screen_derivations(train, features, target)
features = [c for c in features if c not in bad]

ph_tokens = {"?", "-", "NA", "N/A"}
ph_cols = []
for c in features:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        stripped = s.astype(str).str.strip()
        if (stripped.isin(ph_tokens) | (stripped == "")).any():
            ph_cols.append(c)

numeric_cols, categorical_cols, high_card = [], [], []
threshold = max(1, int(0.05 * len(train)))
for c in features:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        dt_tr = pd.to_datetime(train[c], errors="coerce")
        dt_va = pd.to_datetime(val[c], errors="coerce")
        for part in ["hour", "dayofweek", "month", "day", "year"]:
            name = f"{c}_{part}"
            train[name] = getattr(dt_tr.dt, part)
            val[name] = getattr(dt_va.dt, part)
            numeric_cols.append(name)
    elif pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    else:
        is_cat = (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(s)
        )
        if is_cat and s.nunique(dropna=True) <= threshold:
            categorical_cols.append(c)
        elif is_cat:
            high_card.append(c)
        elif s.nunique(dropna=True) <= threshold:
            categorical_cols.append(c)
        else:
            high_card.append(c)

final_features = numeric_cols + categorical_cols
n_out = len(numeric_cols) + sum(max(1, train[c].nunique(dropna=False)) for c in categorical_cols)
sparse_cat = n_out > 300
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_cat)),
])
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
transformers = []
if numeric_cols:
    transformers.append(("num", num_pipe, numeric_cols))
if categorical_cols:
    transformers.append(("cat", cat_pipe, categorical_cols))
preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
if n_out <= 300:
    preprocessor.set_output(transform="pandas")
    num_pipe.set_output(transform="pandas")
    cat_pipe.set_output(transform="pandas")

X_train = train[final_features]
X_val = val[final_features]
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, "train", y=train[target])
Xt = preprocessor.transform(X_val)

mlca_runtime.record_metric("n_features", float(len(final_features)), "train")
mlca_runtime.record_metric("n_train_rows", float(len(train)), "train")
mlca_runtime.record_metric("n_val_rows", float(len(val)), "val")
mlca_runtime.record_metric("n_output_columns", float(Xt.shape[1]), "val")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

print(f"features={len(final_features)} numeric={len(numeric_cols)} categorical={len(categorical_cols)} high_card_dropped={len(high_card)}")
print(f"placeholder_cols={len(ph_cols)} excluded_derivations={len(bad)}")
if bad:
    ex = "; ".join(f"{c}: {r}" for c, r in list(bad.items())[:2])
    print(f"excluded {ex}")
print(f"transformed train_rows={len(X_train)} val_rows={len(X_val)} output_cols={Xt.shape[1]}")
