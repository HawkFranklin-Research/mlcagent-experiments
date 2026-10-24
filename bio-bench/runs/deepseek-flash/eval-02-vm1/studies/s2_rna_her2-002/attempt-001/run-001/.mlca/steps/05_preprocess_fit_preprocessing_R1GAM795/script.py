import mlca_runtime, json, re, itertools
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])
proto = json.loads(paths["_protocol"].read_text())

def need(k):
    if k not in proto:
        raise KeyError(f"Missing {k}; available: {list(proto.keys())}")
    return proto[k]

targets = list(need("targets"))
excluded = need("excluded_columns")
if not isinstance(excluded, (list, tuple, set)):
    raise TypeError(f"excluded_columns must be a list; available top-level: {list(proto.keys())}")
excluded = set(excluded)
group = need("group_column")
_ = need("task")
target = targets[0]
if target not in train.columns:
    raise KeyError(target)

cand = [c for c in train.columns if c not in set(targets) | excluded | {group}]

def date_like(s):
    vals = s.dropna().astype(str).head(20)
    return len(vals) > 0 and all(re.match(r"^\d{4}-\d{2}-\d{2}", v) for v in vals)

new_dates = []
for c in list(cand):
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s) or (
        (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and date_like(s)
    ):
        dt = pd.to_datetime(train[c], errors="coerce")
        dv = pd.to_datetime(val[c], errors="coerce")
        for p in ("hour", "dayofweek", "month"):
            n = f"{c}_{p}"
            train[n] = getattr(dt.dt, p)
            val[n] = getattr(dv.dt, p)
            new_dates.append(n)
        cand.remove(c)
cand.extend(new_dates)

for c in list(cand):
    s = train[c]
    if s.nunique(dropna=True) == len(train):
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s):
            cand.remove(c)

y = train[target]
removed = {}
for c in cand:
    s = train[c]
    if s.notna().all() and y.notna().all() and s.eq(y).all():
        removed[c] = "equals target"

if pd.api.types.is_numeric_dtype(y):
    u = set(pd.unique(y.dropna()))
    if u.issubset({0, 1}) and (y == 0).any() and (y == 1).any():
        pos = y == 1
        cand2 = [c for c in cand if c not in removed and pd.api.types.is_numeric_dtype(train[c])
                 and train[c].notna().all() and (train[c] >= 0).all()
                 and (train.loc[~pos, c] == 0).all()]
        for c in cand2:
            if np.allclose(train.loc[pos, c].to_numpy(), 1.0):
                removed[c] = "sums to target"
        sig = {tuple(np.round(train.loc[pos, c].to_numpy(), 12)): c for c in cand2 if c not in removed}
        for c in cand2:
            if c in removed:
                continue
            comp = tuple(np.round(1.0 - train.loc[pos, c].to_numpy(), 12))
            if comp in sig and sig[comp] != c:
                removed[c] = removed[sig[comp]] = "sums to target"
        left = [c for c in cand2 if c not in removed]
        if len(left) <= 150:
            for a, b, d in itertools.combinations(left, 3):
                if np.allclose(train.loc[pos, a].to_numpy() + train.loc[pos, b].to_numpy() + train.loc[pos, d].to_numpy(), 1.0):
                    removed[a] = removed[b] = removed[d] = "sums to target"
                    break

cand = [c for c in cand if c not in removed]
numeric_cols, categorical_cols, dropped_high = [], [], []
for c in cand:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    elif s.nunique(dropna=True) > 0.05 * len(train):
        dropped_high.append(c)
    else:
        categorical_cols.append(c)
feature_cols = numeric_cols + categorical_cols
if not feature_cols:
    raise ValueError("No feature columns available after exclusions")

warn = []
if pd.api.types.is_numeric_dtype(y):
    corr = train[numeric_cols].corrwith(y).abs()
    warn = corr[corr >= 0.98].index.tolist()

tokens = {"?", "-", "NA", "N/A"}
ph_cols, ph_tokens = [], set()
for c in categorical_cols:
    s = train[c].astype("string").str.strip()
    found = set(s[s.isin(tokens)].unique())
    if s.eq("").any():
        found.add("")
    if found:
        ph_cols.append(c)
        ph_tokens |= found

wide = len(numeric_cols) + sum(train[c].nunique(dropna=True) for c in categorical_cols) > 300
cat_steps = [
    ("impute_nan", SimpleImputer(strategy="most_frequent")),
    ("strip", FunctionTransformer(np.char.strip, feature_names_out="one-to-one")),
]
for i, tok in enumerate(sorted(ph_tokens)):
    cat_steps.append((f"impute_tok_{i}", SimpleImputer(missing_values=tok, strategy="most_frequent")))
cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=wide)))

trans = []
if numeric_cols:
    trans.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_cols))
if categorical_cols:
    trans.append(("cat", Pipeline(cat_steps), categorical_cols))
pre = ColumnTransformer(trans, remainder="drop", sparse_threshold=1.0 if wide else 0.3)
if not wide:
    pre.set_output(transform="pandas")
Xtr, Xv = train[feature_cols], val[feature_cols]
pre = mlca_runtime.fit_record(pre, Xtr, "train", y=y)
Xt = pre.transform(Xv)

if removed:
    print("Derivation-excluded:", ", ".join(f"{k} ({v})" for k, v in list(removed.items())[:5]))
if dropped_high:
    print("High-cardinality dropped:", ", ".join(dropped_high[:10]))
if ph_cols:
    print("Placeholders in text cols:", len(ph_cols), ", ".join(sorted(ph_tokens)))
if warn:
    print("Correlation >=0.98 kept:", ", ".join(warn[:10]))
print("Features:", len(feature_cols), "transformed val:", Xt.shape)
mlca_runtime.save_model(pre, "preprocessor.pkl")
