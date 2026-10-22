import mlca_runtime
import json, re
from functools import partial
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
drop = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in drop]

keep = []
for c in feature_cols:
    s = train[c]
    if s.isna().all():
        keep.append(c)
        continue
    is_text = pd.api.types.is_object_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s.dtype)
    if is_text and s.nunique(dropna=True) == s.notna().sum() == len(s):
        continue
    if pd.api.types.is_integer_dtype(s.dtype) and len(s) > 1 and s.nunique(dropna=True) == len(s):
        vals = np.sort(s.dropna().unique())
        seq = bool(np.all(np.diff(vals) == 1)) if len(vals) > 1 else False
        name = c.lower()
        if seq or any(tok in name for tok in ("id", "code", "key", "barcode", "sample", "patient")):
            continue
    keep.append(c)
feature_cols = keep

def dateish(s):
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return True
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        sample = s.dropna().head(20)
        if len(sample):
            try:
                return bool(pd.to_datetime(sample, errors="coerce").notna().all())
            except Exception:
                return False
    return False

date_cols = [c for c in feature_cols if dateish(train[c])]
if date_cols:
    ref = pd.Timestamp("1970-01-01")
    for df in (train, val):
        for c in date_cols:
            df[c] = (pd.to_datetime(df[c], errors="coerce") - ref).dt.days.astype(float)

ph_pat = re.compile(r"^\s*(\?|-|NA|N/A)\s*$|^\s+$", re.I)
ph_cols = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s.dtype):
        if s.astype("string").str.match(ph_pat).any():
            ph_cols.append(c)
print(f"Placeholders found: ?,-,NA,N/A,whitespace in {len(ph_cols)} columns")

def derived_features(X, y, cols):
    found = set()
    for c in cols:
        s = X[c]
        if ((s == y) | (s.isna() & y.isna())).all():
            found.add(c)
    if not pd.api.types.is_numeric_dtype(y.dtype):
        return found
    yv = y.to_numpy(dtype=float)
    cand = []
    for c in cols:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s.dtype):
            v = s.to_numpy(dtype=float)
            if np.all(np.isfinite(v)) and np.all(v >= 0) and np.all(v <= yv + 1e-12):
                cand.append(c)
    sigmap = {X[c].to_numpy(dtype=float).tobytes(): c for c in cand}
    for c in cand:
        res = yv - X[c].to_numpy(dtype=float)
        if np.all(res >= -1e-12):
            hit = sigmap.get(res.tobytes())
            if hit and hit != c:
                found.add(c)
                found.add(hit)
    if len(cand) <= 200:
        for i, ci in enumerate(cand):
            xi = X[ci].to_numpy(dtype=float)
            for cj in cand[i:]:
                res = yv - xi - X[cj].to_numpy(dtype=float)
                if np.all(res >= -1e-12):
                    hit = sigmap.get(res.tobytes())
                    if hit and hit not in (ci, cj):
                        found.update([ci, cj, hit])
    return found

excluded_derived = set()
corr_warn = set()
for t in targets:
    if t not in train.columns:
        continue
    y = train[t]
    excluded_derived |= derived_features(train, y, feature_cols)
    if pd.api.types.is_numeric_dtype(y.dtype) and y.nunique(dropna=True) > 10:
        num = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c].dtype)]
        if num:
            corr = train[num].corrwith(y).abs()
            corr_warn.update(corr[corr >= 0.98].index.tolist())

feature_cols = [c for c in feature_cols if c not in excluded_derived]
if excluded_derived:
    print(f"Excluded derivations (target identity/sum): {sorted(excluded_derived)}")
else:
    print("Excluded derivations: none")
if corr_warn:
    print(f"Correlation >=0.98 warnings: {sorted(corr_warn)}")

num_cols, cat_cols, high_card = [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_bool_dtype(s.dtype):
        train[c] = s.astype(int)
        val[c] = val[c].astype(int)
        num_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s.dtype):
        num_cols.append(c)
    else:
        n = s.nunique(dropna=True)
        if n <= 0.05 * len(train):
            cat_cols.append(c)
        else:
            high_card.append(c)

model_cols = [c for c in feature_cols if c not in high_card]
width = len(num_cols) + sum(train[c].nunique(dropna=False) for c in cat_cols)
sparse = width > 300

clean = FunctionTransformer(partial(
    pd.DataFrame.replace,
    to_replace={r"^\s+|\s+$": "", r"^\s*(\?|-|NA|N/A)?\s*$": np.nan},
    regex=True,
))
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
cat_pipe = Pipeline([
    ("clean", clean),
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)),
])

transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
preprocessor = ColumnTransformer(transformers, remainder="drop")
if not sparse:
    preprocessor.set_output(transform="pandas")

preprocessor = mlca_runtime.fit_record(preprocessor, train[model_cols], "train")
Xt_val = preprocessor.transform(val[model_cols])
print(f"Val transformed shape: {Xt_val.shape}")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
