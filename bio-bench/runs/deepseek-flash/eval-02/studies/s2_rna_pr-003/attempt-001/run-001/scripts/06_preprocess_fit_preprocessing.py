import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_string_dtype

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
protocol = json.loads(inp["_protocol"].read_text())
for key in ("targets", "excluded_columns"):
    if key not in protocol:
        raise KeyError(f"Missing {key}; available keys: {list(protocol)}")
targets = list(protocol["targets"])
target = targets[0]
excluded = set(protocol["excluded_columns"])
group_col = protocol.get("group_column")
if group_col:
    excluded.add(group_col)

cand = [c for c in train.columns if c not in set(targets) and c not in excluded and c != group_col]
feat = []
for c in cand:
    s = train[c]
    if (s.dtype == object or is_string_dtype(s.dtype)) and s.nunique(dropna=True) == len(train):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(train) and ("id" in c.lower() or "code" in c.lower()):
        continue
    feat.append(c)

ph = {}
for c in feat:
    s = train[c]
    if s.dtype == object or is_string_dtype(s.dtype):
        s_str = s.astype("string").str.strip()
        mask = s_str.isin(["?", "-", "NA", "N/A"]) | s_str.eq("")
        for token in ["?", "-", "NA", "N/A", ""]:
            if (s_str[mask] == token).any():
                ph.setdefault(token, set()).add(c)
        for df in (train, val):
            ss = df[c].astype("string").str.strip()
            m = ss.isin(["?", "-", "NA", "N/A"]) | ss.eq("")
            df[c] = ss.mask(m, pd.NA).astype(object)
print("placeholders:", "none" if not ph else "; ".join(f"{t!r} in {len(v)} cols" for t, v in ph.items()))

final_feat = []
for c in feat:
    s = train[c]
    is_dt = is_datetime64_any_dtype(s)
    if not is_dt and (s.dtype == object or is_string_dtype(s.dtype)):
        p = pd.to_datetime(s, errors="coerce")
        if p.notna().mean() > 0.9:
            is_dt = True
            train[c] = p
            val[c] = pd.to_datetime(val[c], errors="coerce")
    if is_dt:
        for suffix, attr in (("month", "month"), ("dow", "dayofweek"), ("hour", "hour")):
            name = f"{c}_{suffix}"
            train[name] = getattr(train[c].dt, attr)
            val[name] = getattr(val[c].dt, attr)
            final_feat.append(name)
    else:
        final_feat.append(c)
feat = final_feat

numeric, categorical = [], []
for c in feat:
    s = train[c]
    if s.notna().sum() == 0:
        continue
    if s.dtype == object or is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=True) <= max(1, 0.05 * len(train)):
            categorical.append(c)
    elif is_numeric_dtype(s):
        numeric.append(c)
feat = numeric + categorical

excluded_features = {}
y_str = train[target].astype("string").fillna("__MISSING__")
for c in list(feat):
    if train[c].astype("string").fillna("__MISSING__").eq(y_str).all():
        excluded_features[c] = "exact equality with target"
if is_numeric_dtype(train[target]) and train[target].nunique(dropna=True) <= 2:
    yv = train[target].to_numpy(dtype=float)
    if set(np.unique(yv)).issubset({0.0, 1.0}):
        pos, neg = yv == 1, yv == 0
        csum = []
        for c in feat:
            if c in excluded_features:
                continue
            x = pd.to_numeric(train[c], errors="coerce").to_numpy(dtype=float)
            if np.isnan(x).any():
                continue
            if np.all(x >= -1e-12) and np.allclose(x[neg], 0, atol=1e-12) and np.all(x[pos] <= 1 + 1e-12):
                csum.append((c, x))
        d = {tuple(np.round(x[pos], 12)): c for c, x in csum}
        found = set()
        for c, x in csum:
            c2 = d.get(tuple(np.round(1 - x[pos], 12)))
            if c2 is not None and c2 != c:
                found.update([c, c2])
        if not found and len(csum) <= 200:
            pair = {}
            for i, (c1, x1) in enumerate(csum):
                for c2, x2 in csum[i + 1:]:
                    pair[tuple(np.round(x1[pos] + x2[pos], 12))] = (c1, c2)
            for c, x in csum:
                p = pair.get(tuple(np.round(1 - x[pos], 12)))
                if p:
                    found.update(p)
                    found.add(c)
        for c in found:
            excluded_features[c] = "row sum of up to 3 equals target"
print("excluded derivations:", "none" if not excluded_features else "; ".join(f"{k} ({v})" for k, v in list(excluded_features.items())[:5]))

numeric = [c for c in numeric if c not in excluded_features]
categorical = [c for c in categorical if c not in excluded_features]
feat = numeric + categorical
width_est = len(numeric) + sum(train[c].nunique(dropna=True) for c in categorical)

transformers = []
if numeric:
    transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric))
if categorical:
    transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=(width_est > 300)))]), categorical))
if not transformers:
    raise ValueError("No usable feature columns after screening")
ct = ColumnTransformer(transformers, remainder="drop")
if width_est <= 300:
    ct.set_output(transform="pandas")
pre = ct if width_est <= 300 else Pipeline([("pre", ct)])
pre = mlca_runtime.fit_record(pre, train[feat], "train", y=train[target])
Xt = pre.transform(val[feat])
print(f"features: {len(numeric)} numeric, {len(categorical)} categorical; encoded width ~{width_est}")
print("val transformed:", Xt.shape)
mlca_runtime.save_model(pre, "preprocessor.pkl")
