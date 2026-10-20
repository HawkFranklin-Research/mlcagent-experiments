import mlca_runtime
import json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_object_dtype, is_string_dtype
from pandas.api.types import CategoricalDtype

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for k in ["targets", "excluded_columns", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing {k}; available top-level keys: {list(protocol)}")
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns") or [])
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing primary_metric; available task keys: {list(task)}")
group = protocol.get("group_column")
train = pd.read_parquet(inp["train"])

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
kept = []
for c in feature_cols:
    s = train[c]
    if (is_object_dtype(s) or is_string_dtype(s)) and s.nunique(dropna=True) == len(train):
        continue
    if c.lower() in {"id", "row_id", "key", "code", "index"} and s.nunique(dropna=True) == len(train):
        continue
    kept.append(c)
feature_cols = kept

date_cols = [c for c in feature_cols if is_datetime64_any_dtype(train[c])]
if date_cols:
    feature_cols = [c for c in feature_cols if c not in date_cols]
    print(f"datetime columns dropped without parsing: {date_cols}")

tokens = {"?", "-", "NA", "N/A"}
ph_cols = {}
for c in feature_cols:
    s = train[c]
    if not (is_object_dtype(s) or is_string_dtype(s)):
        continue
    found = set()
    for v in s.dropna().astype(str).unique():
        t = v.strip()
        if t in tokens or t == "":
            found.add(t if t else "<whitespace>")
    if found:
        ph_cols[c] = found
if ph_cols:
    all_ph = sorted({p for v in ph_cols.values() for p in v})
    print(f"placeholders found {all_ph} in {len(ph_cols)} columns")
else:
    print("placeholders found none")

targets_present = [t for t in targets if t in train.columns]
excl_deriv = set()
for t in targets_present:
    y = train[t].fillna(-999999)
    for f in feature_cols:
        if f in excl_deriv:
            continue
        if train[f].equals(train[t]) or (train[f].fillna(-999999) == y).all():
            excl_deriv.add(f)
            print(f"excluded derived feature {f} == {t}")

nonneg = [f for f in feature_cols if f not in excl_deriv and is_numeric_dtype(train[f]) and (train[f] >= 0).all()]
for t in targets_present:
    y = train[t]
    found = False
    for r in range(1, 4):
        for combo in combinations(nonneg, r):
            if any(f in excl_deriv for f in combo):
                continue
            s = train[list(combo)].sum(axis=1)
            if (s == y).all():
                for f in combo:
                    if f not in excl_deriv:
                        excl_deriv.add(f)
                        print(f"excluded derived feature {f} in sum == {t}")
                found = True
                break
        if found:
            break
feature_cols = [f for f in feature_cols if f not in excl_deriv]

warn = set()
for t in targets_present:
    if not is_numeric_dtype(train[t]) or train[t].nunique(dropna=True) < 2:
        continue
    for f in feature_cols:
        if not is_numeric_dtype(train[f]):
            continue
        corr = train[f].corr(train[t])
        if pd.notna(corr) and abs(corr) >= 0.98:
            warn.add(f)
print("corr>=0.98 kept: " + (", ".join(sorted(warn)) if warn else "none"))

num_cols, cat_cols, drop_cols = [], [], []
for c in feature_cols:
    s = train[c]
    if is_numeric_dtype(s):
        num_cols.append(c)
    elif is_object_dtype(s) or is_string_dtype(s) or isinstance(s.dtype, CategoricalDtype):
        if s.nunique(dropna=True) <= max(5, int(0.05 * len(train))):
            cat_cols.append(c)
        else:
            drop_cols.append(c)
    else:
        cat_cols.append(c)
if drop_cols:
    print(f"high-cardinality text dropped: {len(drop_cols)}")

width = len(num_cols) + sum(train[c].nunique(dropna=True) + len(ph_cols.get(c, set())) for c in cat_cols)
dense = width <= 300
transformers = []
if num_cols:
    transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols))
for c in cat_cols:
    steps = [("imp_nan", SimpleImputer(strategy="most_frequent")), ("strip", FunctionTransformer(np.char.strip))]
    for token in sorted(ph_cols.get(c, set())):
        steps.append((f"imp_{len(steps)}", SimpleImputer(missing_values="" if token == "<whitespace>" else token, strategy="most_frequent")))
    steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)))
    transformers.append((f"cat_{c}", Pipeline(steps), [c]))

ct = ColumnTransformer(transformers, remainder="drop")
if dense:
    ct.set_output(transform="pandas")
fitted = mlca_runtime.fit_record(ct, train[feature_cols], "train")
mlca_runtime.save_model(fitted, "preprocessor.pkl")
mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")
print(f"train rows {len(train)} features {len(feature_cols)} saved preprocessor.pkl")
