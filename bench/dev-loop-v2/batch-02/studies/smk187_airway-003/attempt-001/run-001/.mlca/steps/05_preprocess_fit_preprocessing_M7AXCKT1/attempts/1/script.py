import mlca_runtime
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
protocol = json.loads(inp["_protocol"].read_text())

for k in ("targets", "excluded_columns", "group_column", "task"):
    if k not in protocol:
        raise ValueError(f"Missing protocol key {k}; available keys {list(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]
task = protocol["task"]
if not targets or targets[0] not in train.columns:
    raise ValueError("Declared target not found in training table")
target = targets[0]

# Placeholder-token check for text columns.
text_like = [c for c in train.columns if c not in targets + excluded + [group]
             and (pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])
                  or isinstance(train[c].dtype, pd.CategoricalDtype))]
ph_cols = []
tokens = {"?", "-", "NA", "N/A", ""}
for c in text_like:
    s = train[c].astype("string")
    mask = s.str.strip().isin(list(tokens)) | s.str.fullmatch(r"\s+")
    if mask.fillna(False).any():
        ph_cols.append(c)
for c in ph_cols:
    for df in (train, val):
        s = df[c].astype("string").str.strip()
        s = s.mask(s.isin(list(tokens)) | s.str.fullmatch(r"\s+"), np.nan)
        df[c] = s
print(f"placeholder columns: {len(ph_cols)}")

def key_like(s, name):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s) and s.is_unique:
        vals = np.sort(s.dropna().unique())
        if len(vals) > 1 and np.all(np.diff(vals) == 1):
            return True
        return any(k in str(name).lower() for k in ("id", "key", "code", "index"))
    return False

base_excluded = set(targets + excluded + ([group] if group in train.columns else []))
feature_cols = [c for c in train.columns if c not in base_excluded and not key_like(train[c], c)]

def screen_derived(X, y, cols):
    derived = {}
    yn = pd.to_numeric(y, errors="coerce")
    if not yn.notna().all():
        return derived
    yv = yn.to_numpy(dtype=float)
    for c in cols:
        if pd.api.types.is_numeric_dtype(X[c]) and np.array_equal(X[c].to_numpy(dtype=float), yv):
            derived[c] = "equals target"
    cand = []
    for c in cols:
        if c in derived:
            continue
        s = X[c]
        if pd.api.types.is_numeric_dtype(s) and s.notna().all():
            v = s.to_numpy(dtype=float)
            if (v >= 0).all() and (v <= yv).all():
                cand.append(c)
    vec = {c: X[c].to_numpy(dtype=float) for c in cand}
    keys = {v.tobytes(): c for c, v in vec.items()}
    for c in cand:
        rem = yv - vec[c]
        if (rem >= 0).all():
            d = keys.get(rem.tobytes())
            if d is not None and d != c:
                derived[c] = f"sum with {d}"
                derived[d] = f"sum with {c}"
    active = [c for c in cand if c not in derived and np.any(vec[c] != 0)]
    if len(active) <= 300:
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                rem = yv - vec[a] - vec[b]
                if (rem >= 0).all():
                    e = keys.get(rem.tobytes())
                    if e is not None and e not in (a, b):
                        for z in (a, b, e):
                            derived[z] = "sum of up to 3"
    return derived

derived = screen_derived(train[feature_cols], train[target], feature_cols)
kept_cols = [c for c in feature_cols if c not in derived]
if derived:
    shown = ", ".join(f"{c}({derived[c]})" for c in list(derived)[:5])
    print(f"derivation exclusions: {len(derived)} [{shown}]")
else:
    print("derivation exclusions: none")

num_cols, cat_cols, drop_cols = [], [], []
for c in kept_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        drop_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        u = s.nunique(dropna=True)
        if u <= max(1, int(0.05 * len(train))):
            cat_cols.append(c)
        else:
            drop_cols.append(c)

width_est = len(num_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
wide = width_est > 300
num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=wide)),
])
transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
if not transformers:
    raise ValueError("No usable feature columns after exclusions")
ct = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
if not wide:
    ct.set_output(transform="pandas")
    num_pipe.set_output(transform="pandas")
    cat_pipe.set_output(transform="pandas")

ct = mlca_runtime.fit_record(ct, train[kept_cols], "train")
n_val, out_width = 0, None
for start in range(0, len(val), 2000):
    z = ct.transform(val[kept_cols].iloc[start:start + 2000])
    n_val += z.shape[0]
    out_width = z.shape[1]
print(f"val transformed: {n_val} rows x {out_width} cols")
mlca_runtime.record_metric("n_features", float(len(kept_cols)), "train")
mlca_runtime.record_metric("n_val_rows_transformed", float(n_val), "val")
mlca_runtime.save_model(ct, "preprocessor.pkl")
print(f"saved preprocessor: numeric={len(num_cols)} categorical={len(cat_cols)} dropped={len(drop_cols)}")
