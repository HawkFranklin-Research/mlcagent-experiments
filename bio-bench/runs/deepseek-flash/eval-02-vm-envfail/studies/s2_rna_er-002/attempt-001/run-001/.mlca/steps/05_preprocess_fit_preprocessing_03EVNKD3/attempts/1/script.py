import json
import numpy as np
import pandas as pd
import mlca_runtime

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
protocol = json.loads(inp["_protocol"].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {list(d)}")
    return d[k]

targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
if isinstance(excluded, dict):
    raise TypeError(f"excluded_columns must be a list; got dict keys {list(excluded)}")
group = protocol.get("group_column")
task_type = protocol.get("task", {}).get("type")
target = targets[0]
y = train[target]

def is_text(s):
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)

def is_key_like(s):
    if pd.api.types.is_datetime64_any_dtype(s) or not s.notna().all():
        return False
    if pd.api.types.is_numeric_dtype(s):
        return pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s)
    if is_text(s):
        return s.nunique(dropna=False) == len(s)
    return False

excl = set(targets) | set(excluded)
if group:
    excl.add(group)
input_cols = [c for c in train.columns if c not in excl and not is_key_like(train[c])]
if not input_cols:
    raise ValueError("No candidate feature columns remain")

derived = {}
yv = y.to_numpy()
for c in input_cols:
    s = train[c]
    if s.notna().all():
        try:
            if (s.to_numpy() == yv).all():
                derived[c] = "equals target"
        except Exception:
            pass

if pd.api.types.is_numeric_dtype(y):
    vals = set(pd.unique(y.dropna()))
    if vals.issubset({0, 1}) and len(vals) <= 2:
        pos, neg = yv == 1, yv == 0
        full = (1 << int(pos.sum())) - 1
        elig = []
        for c in input_cols:
            if c in derived:
                continue
            s = train[c]
            if not pd.api.types.is_numeric_dtype(s):
                continue
            arr = s.to_numpy(dtype=float)
            if np.isnan(arr).any() or (arr < 0).any() or (arr > 1).any() or (arr[neg] != 0).any():
                continue
            m = 0
            for i, v in enumerate(arr[pos]):
                if v > 0.5:
                    m |= 1 << i
            if m:
                elig.append((m, c))
        mask_to_col = {}
        for m, c in elig:
            mask_to_col.setdefault(m, c)
        for m, c in elig:
            if m == full:
                derived[c] = "sum equals target"
                break
        if not derived:
            for m, c in elig:
                c2 = mask_to_col.get(full ^ m)
                if c2 is not None and c2 != c:
                    derived[c] = derived[c2] = "sum equals target"
                    break
        if not derived and len(elig) <= 80:
            for i in range(len(elig)):
                for j in range(i + 1, len(elig)):
                    m1, c1 = elig[i]
                    m2, c2 = elig[j]
                    if m1 & m2:
                        continue
                    c3 = mask_to_col.get(full ^ (m1 | m2))
                    if c3 is not None and c3 not in (c1, c2):
                        for cc in (c1, c2, c3):
                            derived[cc] = "sum equals target"
                        break
                if derived:
                    break

input_cols = [c for c in input_cols if c not in derived]

corr_warn = []
if task_type != "binary" and pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
    num = train[input_cols].select_dtypes(include=[np.number])
    cor = num.corrwith(y).abs()
    corr_warn = list(cor[cor >= 0.98].index)

placeholder_cols = []
for c in input_cols:
    s = train[c]
    if is_text(s):
        ss = s.dropna().astype(str)
        st = ss.str.strip()
        if ((ss != st) | st.isin({"?", "-", "NA", "N/A", ""})).any():
            placeholder_cols.append(c)
            for df in (train, val):
                if c in df:
                    x = df[c].astype("string").str.strip()
                    df[c] = x.replace({"?": pd.NA, "-": pd.NA, "NA": pd.NA, "N/A": pd.NA, "": pd.NA})

numeric_cols, categorical_cols, high_card, final_input_cols = [], [], [], []
for c in input_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        for df in (train, val):
            if c in df:
                for suffix, vals in (("_month", df[c].dt.month), ("_dow", df[c].dt.dayofweek), ("_hour", df[c].dt.hour)):
                    df[c + suffix] = vals
                numeric_cols.append(c + "_month")
                numeric_cols.append(c + "_dow")
                numeric_cols.append(c + "_hour")
                final_input_cols.extend([c + "_month", c + "_dow", c + "_hour"])
        continue
    if is_text(s):
        parsed = pd.to_datetime(s, errors="coerce")
        if parsed.notna().sum() and parsed.notna().mean() >= 0.9:
            for df in (train, val):
                if c in df:
                    p = pd.to_datetime(df[c], errors="coerce")
                    df[c + "_month"], df[c + "_dow"], df[c + "_hour"] = p.dt.month, p.dt.dayofweek, p.dt.hour
            numeric_cols.extend([c + "_month", c + "_dow", c + "_hour"])
            final_input_cols.extend([c + "_month", c + "_dow", c + "_hour"])
            continue
    final_input_cols.append(c)
    if pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    else:
        if s.nunique(dropna=True) > max(5, 0.05 * len(train)):
            high_card.append(c)
        else:
            categorical_cols.append(c)

cat_width = sum(max(1, train[c].nunique(dropna=True)) for c in categorical_cols)
dense = len(numeric_cols) + cat_width <= 300
transformers = []
if numeric_cols:
    transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler())]), numeric_cols))
if categorical_cols:
    transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense))]), categorical_cols))
if not transformers:
    raise ValueError("No numeric or categorical feature columns remain")

ct = ColumnTransformer(transformers, remainder="drop")
pre = Pipeline([("ct", ct)])
if dense:
    ct.set_output(transform="pandas")
    pre.set_output(transform="pandas")

X_train, X_val = train[final_input_cols], val[final_input_cols]
pre = mlca_runtime.fit_record(pre, X_train, "train", y=y)
Xt = pre.transform(X_val)
mlca_runtime.save_model(pre, "preprocessor.pkl")

msgs = [f"Features: {len(final_input_cols)} input, {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, {len(high_card)} high-card dropped"]
if derived:
    msgs.append("Derivation exclusions: " + "; ".join(f"{c} ({r})" for c, r in derived.items()))
if corr_warn:
    msgs.append("High target corr >=0.98: " + ", ".join(map(str, corr_warn)))
if placeholder_cols:
    msgs.append(f"Placeholders found: ?,-,NA,N/A; columns={len(placeholder_cols)}")
msgs.append(f"Val transform shape: {Xt.shape}")
for m in msgs[:5]:
    print(m)
