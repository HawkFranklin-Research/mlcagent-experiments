import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in proto:
        raise KeyError(f"Missing {key}; available keys: {list(proto)}")

targets = list(proto["targets"])
excluded = set(proto["excluded_columns"])
group_col = proto.get("group_column")
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
target = targets[0]
y = train[target]

drop = set(targets) | excluded
if group_col:
    drop.add(group_col)
cand = [c for c in train.columns if c not in drop]

placeholders = {"?", "-", "NA", "N/A"}
ph_cols = []
for c in list(cand):
    if pd.api.types.is_numeric_dtype(train[c]) or pd.api.types.is_datetime64_any_dtype(train[c]):
        continue
    s = train[c].astype("string").str.strip()
    mask = s.isin(placeholders) | s.eq("")
    if mask.any():
        ph_cols.append(c)
    cleaned = s.mask(mask)
    train[c] = cleaned.astype(object).where(cleaned.notna(), np.nan)
    sv = val[c].astype("string").str.strip()
    vm = sv.isin(placeholders) | sv.eq("")
    vc = sv.mask(vm)
    val[c] = vc.astype(object).where(vc.notna(), np.nan)

for c in list(cand):
    if pd.api.types.is_datetime64_any_dtype(train[c]):
        dt = pd.to_datetime(train[c])
        for part in ("year", "month", "day", "dayofweek", "hour"):
            nc = f"{c}_{part}"
            train[nc] = getattr(dt.dt, part)
            val[nc] = getattr(pd.to_datetime(val[c]).dt, part)
            cand.append(nc)
        train.drop(columns=[c], inplace=True)
        val.drop(columns=[c], inplace=True)
        cand.remove(c)

deriv = set()
y_num = pd.to_numeric(y, errors="coerce")
for c in cand:
    if pd.api.types.is_numeric_dtype(train[c]) and pd.api.types.is_numeric_dtype(y):
        same = train[c] == y
    else:
        same = train[c].astype("string") == y.astype("string")
    if train[c].notna().all() and same.all():
        deriv.add(c)

if y_num.notna().all() and set(y_num.unique()).issubset({0, 1}):
    zero = y_num == 0
    comps = []
    for c in cand:
        v = pd.to_numeric(train[c], errors="coerce")
        if v.notna().all() and (v >= 0).all() and (v[zero] == 0).all():
            comps.append((c, v))
    if len(comps) <= 30:
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                s2 = comps[i][1] + comps[j][1]
                if (s2 == y_num).all():
                    deriv.update([comps[i][0], comps[j][0]])
                for k in range(j + 1, len(comps)):
                    if (s2 + comps[k][1] == y_num).all():
                        deriv.update([comps[i][0], comps[j][0], comps[k][0]])

features = [c for c in cand if c not in deriv]
warn = []
if pd.api.types.is_numeric_dtype(y):
    num_corr = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    if num_corr and y.notna().all():
        corr = train[num_corr].corrwith(y.astype(float)).abs()
        warn = [c for c in num_corr if corr[c] >= 0.98]

n = len(train)
num_cols, cat_cols = [], []
for c in features:
    if pd.api.types.is_numeric_dtype(train[c]) or pd.api.types.is_bool_dtype(train[c]):
        num_cols.append(c)
    elif train[c].nunique(dropna=True) <= 0.05 * n:
        cat_cols.append(c)

width = len(num_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
dense = width <= 300
trans = []
if num_cols:
    trans.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols))
if cat_cols:
    trans.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense))]), cat_cols))
if not trans:
    raise ValueError("No usable feature columns")

pre = ColumnTransformer(trans, remainder="drop", sparse_threshold=1.0 if not dense else 0.3)
if dense:
    pre.set_output(transform="pandas")
pre = mlca_runtime.fit_record(pre, train[num_cols + cat_cols], "train", y=y)
Xt = pre.transform(val[num_cols + cat_cols])
mlca_runtime.save_model(pre, "preprocessor.pkl")

print(f"features={len(num_cols) + len(cat_cols)} numeric={len(num_cols)} categorical={len(cat_cols)}")
print(f"placeholder_cols={len(ph_cols)} excluded_derivations={len(deriv)}")
if warn:
    print("high_corr_features=" + ",".join(warn[:10]))
print(f"val_transformed_shape={getattr(Xt, 'shape', None)}")
