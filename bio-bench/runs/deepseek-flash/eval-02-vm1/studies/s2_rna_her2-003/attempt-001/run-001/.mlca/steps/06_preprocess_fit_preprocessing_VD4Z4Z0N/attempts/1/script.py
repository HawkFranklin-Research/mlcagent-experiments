import json
import numpy as np
import pandas as pd
import mlca_runtime
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths["_protocol"].read_text())
for k in ("targets", "excluded_columns", "group_column", "task"):
    if k not in proto:
        raise KeyError(f"Missing protocol key {k!r}; available: {sorted(proto)}")
targets, excluded, group = proto["targets"], proto["excluded_columns"], proto["group_column"]
if isinstance(excluded, dict):
    for k in ("columns", "excluded_columns", "full_list"):
        if isinstance(excluded.get(k), list):
            excluded = excluded[k]
            break
    else:
        raise KeyError(f"excluded_columns has no full list; keys={sorted(excluded)}")
exclude_set = set(targets) | set(excluded)
if group is not None:
    exclude_set.add(group)
train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

feature_cols = []
for c in [c for c in train.columns if c not in exclude_set]:
    s = train[c]
    is_text = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)
    if is_text and s.nunique(dropna=True) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.nunique(dropna=True) == len(s):
        continue
    feature_cols.append(c)

y = train[targets[0]]
deriv = set()
for c in feature_cols:
    s = train[c]
    if (s.eq(y) | (s.isna() & y.isna())).all():
        deriv.add(c)
if pd.api.types.is_numeric_dtype(y) and y.notna().all():
    yv = y.to_numpy()
    num_c = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_bool_dtype(train[c])
             and train[c].notna().all() and (train[c] >= 0).all()]
    uniq = np.unique(yv)
    if len(uniq) <= 2 and all(v in (0, 1) for v in uniq):
        pos, neg = yv == 1, yv == 0
        cands = [c for c in num_c if (train[c].to_numpy()[neg] == 0).all() and (train[c].to_numpy()[pos] <= 1).all()]
        for r in (1, 2, 3):
            for combo in combinations(cands, r):
                if np.array_equal(sum(train[c].to_numpy() for c in combo), yv):
                    deriv.update(combo)
    elif len(num_c) <= 25:
        for r in (1, 2, 3):
            for combo in combinations(num_c, r):
                if np.array_equal(sum(train[c].to_numpy() for c in combo), yv):
                    deriv.update(combo)
if deriv:
    feature_cols = [c for c in feature_cols if c not in deriv]
    print("derivation-excluded:", ", ".join(sorted(deriv)))

warn = []
if pd.api.types.is_numeric_dtype(y):
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train[c]):
            r = train[c].corr(y)
            if pd.notna(r) and abs(r) >= 0.98:
                warn.append(c)
if warn:
    print("correlation>=0.98 kept:", ", ".join(warn[:20]) + ("..." if len(warn) > 20 else ""))

Xtr, Xva = train[feature_cols].copy(), val[feature_cols].copy()
obj = [c for c in feature_cols if pd.api.types.is_object_dtype(Xtr[c]) or pd.api.types.is_string_dtype(Xtr[c]) or isinstance(Xtr[c].dtype, pd.CategoricalDtype)]
ph = []
for c in obj:
    s = Xtr[c].astype("string")
    if s.str.strip().isin(["?", "-", "NA", "N/A"]).any() or s.str.contains(r"^\s*$", regex=True).any() or (s != s.str.strip()).any():
        ph.append(c)
    repl = {"?": pd.NA, "-": pd.NA, "NA": pd.NA, "N/A": pd.NA, "": pd.NA}
    Xtr[c] = s.str.strip().replace(repl)
    Xva[c] = Xva[c].astype("string").str.strip().replace(repl)
if ph:
    print(f"placeholder-cleaned columns: {len(ph)}")

date_cols = []
for c in obj:
    parsed = pd.to_datetime(Xtr[c], errors="coerce")
    if parsed.notna().mean() > 0.8:
        date_cols.append(c)
        pv = pd.to_datetime(Xva[c], errors="coerce")
        ref = parsed.min()
        Xtr[f"{c}_hour"], Xtr[f"{c}_dow"], Xtr[f"{c}_month"] = parsed.dt.hour, parsed.dt.dayofweek, parsed.dt.month
        Xtr[f"{c}_days"] = (parsed - ref).dt.days
        Xva[f"{c}_hour"], Xva[f"{c}_dow"], Xva[f"{c}_month"] = pv.dt.hour, pv.dt.dayofweek, pv.dt.month
        Xva[f"{c}_days"] = (pv - ref).dt.days
if date_cols:
    Xtr, Xva = Xtr.drop(columns=date_cols), Xva.drop(columns=date_cols)
feature_cols = list(Xtr.columns)

num_cols, cat_cols, drop_cols = [], [], []
thr = 0.05 * len(Xtr)
for c in feature_cols:
    s = Xtr[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        (cat_cols if s.nunique(dropna=True) <= thr else drop_cols).append(c)
    elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    else:
        drop_cols.append(c)
if drop_cols:
    Xtr, Xva = Xtr.drop(columns=drop_cols), Xva.drop(columns=drop_cols)
    print("dropped unsupported/high-card:", ", ".join(sorted(drop_cols)))

n_cat_out = sum(Xtr[c].nunique(dropna=True) for c in cat_cols)
width = len(num_cols) + n_cat_out
dense = width <= 300
num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense))])
trans = []
if num_cols:
    trans.append(("num", num_pipe, num_cols))
if cat_cols:
    trans.append(("cat", cat_pipe, cat_cols))
pre = ColumnTransformer(trans, remainder="drop", sparse_threshold=0.3 if dense else 1.0)
if dense:
    pre.set_output(transform="pandas")
    num_pipe.set_output(transform="pandas")
    cat_pipe.set_output(transform="pandas")
pre = mlca_runtime.fit_record(pre, Xtr, "train")
Xt = pre.transform(Xva)
n_out = len(pre.get_feature_names_out())
mlca_runtime.record_metric("preprocessor_output_columns", float(n_out), "train")
mlca_runtime.save_model(pre, "preprocessor.pkl")
print(f"features: {len(feature_cols)} (num={len(num_cols)}, cat={len(cat_cols)}); output={n_out}; val={Xt.shape}")
