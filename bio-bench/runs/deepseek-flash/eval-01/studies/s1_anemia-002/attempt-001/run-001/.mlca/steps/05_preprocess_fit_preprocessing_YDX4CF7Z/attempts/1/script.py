import json, itertools
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
tr = pd.read_parquet(inp["train"])
va = pd.read_parquet(inp["val"])
prot = json.loads(inp["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in prot:
        raise KeyError(f"Missing protocol key {k}; available: {list(prot.keys())}")
targets = prot["targets"]
excluded = prot["excluded_columns"]
group = prot.get("group_column")
target = targets[0]
if target not in tr.columns:
    raise ValueError(f"Target {target} not in train")

base = [c for c in tr.columns if c not in targets and c not in excluded and c != group]

def keylike(s):
    if s.dtype == object or str(s.dtype) == "category":
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        u = np.sort(s.dropna().unique())
        return len(u) == len(s) and len(u) > 1 and u.min() in (0, 1) and u.max() - u.min() + 1 == len(u)
    return False

base = [c for c in base if not keylike(tr[c])]

for c in list(base):
    s = tr[c]
    is_dt = pd.api.types.is_datetime64_any_dtype(s)
    if not is_dt and s.dtype == object:
        p = pd.to_datetime(s, errors="coerce")
        is_dt = p.notna().mean() > 0.9 and s.nunique() > 1
    if is_dt:
        for df in (tr, va):
            dt = pd.to_datetime(df[c], errors="coerce")
            df[c + "_hour"] = dt.dt.hour
            df[c + "_dow"] = dt.dt.dayofweek
            df[c + "_month"] = dt.dt.month
            df.drop(columns=[c], inplace=True)
        base.remove(c)
        base += [c + "_hour", c + "_dow", c + "_month"]

ph = ["?", "-", "NA", "N/A"]
ph_cols = []
for c in base:
    if tr[c].dtype == object:
        s = tr[c]
        mask = s.notna()
        t = s.astype(str).str.strip()
        if (t.isin(ph) | t.eq("")).any():
            ph_cols.append(c)
        t = t.replace(ph, np.nan).replace("", np.nan).where(mask, np.nan)
        tr[c] = t
        va[c] = va[c].astype(str).str.strip().replace(ph, np.nan).replace("", np.nan).where(va[c].notna(), np.nan)
if ph_cols:
    print(f"Placeholders {ph} found in {len(ph_cols)} columns")

drop_der = set()
y = tr[target]
for c in base:
    if tr[c].equals(y):
        drop_der.add(c)
yn = pd.to_numeric(y, errors="coerce")
if yn.notna().all():
    nums = [c for c in base if pd.api.types.is_numeric_dtype(tr[c]) and tr[c].min() >= 0]
    for r in (1, 2, 3):
        for cols in itertools.combinations(nums, r):
            if (tr[list(cols)].sum(axis=1) == yn).all():
                drop_der.update(cols)
if drop_der:
    print(f"Excluded target derivations: {sorted(drop_der)}")

warn = []
if pd.api.types.is_numeric_dtype(y):
    for c in base:
        if c in drop_der or not pd.api.types.is_numeric_dtype(tr[c]):
            continue
        corr = tr[[c, target]].corr().iloc[0, 1]
        if abs(corr) >= 0.98:
            warn.append(c)
if warn:
    print(f"Correlation >=0.98 with target: {warn}")

feat = [c for c in base if c not in drop_der]
num, cat = [], []
n = len(tr)
for c in feat:
    s = tr[c]
    if pd.api.types.is_numeric_dtype(s):
        num.append(c)
    elif s.dtype == object or str(s.dtype) == "category":
        if s.nunique(dropna=True) <= 0.05 * n:
            cat.append(c)
    else:
        num.append(c)

width = len(num) + sum(tr[c].nunique(dropna=True) for c in cat)
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
if width <= 300:
    num_pipe = num_pipe.set_output(transform="pandas")
    cat_pipe = cat_pipe.set_output(transform="pandas")
    trans = []
    if num: trans.append(("num", num_pipe, num))
    if cat: trans.append(("cat", cat_pipe, cat))
    pre = ColumnTransformer(trans, remainder="drop").set_output(transform="pandas")
else:
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    trans = []
    if num: trans.append(("num", num_pipe, num))
    if cat: trans.append(("cat", cat_pipe, cat))
    pre = ColumnTransformer(trans, remainder="drop")

Xtr = tr[feat].copy()
Xva = va[feat].copy()
pre = mlca_runtime.fit_record(pre, Xtr, "train")
Xt = pre.transform(Xva)
print(f"Features {len(feat)} -> encoded width {Xt.shape[1]}")
mlca_runtime.save_model(pre, "preprocessor.pkl")
