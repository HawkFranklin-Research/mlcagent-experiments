import json
import numpy as np
import pandas as pd
import mlca_runtime
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

def required(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {list(d.keys())}")
    return d[k]

targets = required(protocol, "targets")
excluded = required(protocol, "excluded_columns")
group = protocol.get("group_column")
task = protocol.get("task", {})
drop = set(targets) | set(excluded)
if group:
    drop.add(group)
time_cols = set(task.get("time_columns", []))
if task.get("type") == "survival":
    for c in train.columns:
        lc = c.lower()
        if "survival" in lc and ("month" in lc or "time" in lc):
            time_cols.add(c)
drop |= time_cols

n = len(train)
feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and n > 1 and s.nunique(dropna=True) == n:
        continue
    if pd.api.types.is_integer_dtype(s) and n > 1 and s.nunique(dropna=True) == n and ("id" in c.lower() or "code" in c.lower()):
        continue
    feature_cols.append(c)

def screen_derived(df, feats, tname):
    t = df[tname]
    t_num = pd.api.types.is_numeric_dtype(t) and not pd.api.types.is_bool_dtype(t)
    reasons, warns = {}, []
    for c in feats:
        if df[c].equals(t):
            reasons[c] = f"equals {tname}"
    if t_num:
        derived = set(reasons)
        cand = [c for c in feats if c not in derived and pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_bool_dtype(df[c]) and df[c].notna().all()
                and (df[c].dropna() >= 0).all()]
        if len(cand) <= 60:
            for r in (2, 3):
                for combo in combinations(cand, r):
                    if df[list(combo)].sum(axis=1).equals(t):
                        for c in combo:
                            reasons.setdefault(c, f"sum of {r} nonnegative cols equals {tname}")
                        break
        for c in feats:
            if c in reasons or not pd.api.types.is_numeric_dtype(df[c]):
                continue
            r = df[[c, tname]].corr().iloc[0, 1]
            if pd.notna(r) and abs(r) >= 0.98:
                warns.append(c)
    return reasons, warns

derived_reasons, corr_warnings = {}, []
for tname in targets:
    if tname in train.columns:
        dr, cw = screen_derived(train, feature_cols, tname)
        derived_reasons.update(dr)
        corr_warnings.extend(cw)
feature_cols = [c for c in feature_cols if c not in derived_reasons]

num_cols, cat_cols, dropped_wide = [], [], []
card_limit = max(20, int(0.05 * n))
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        dropped_wide.append(c)
    elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    elif pd.api.types.is_bool_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=True) > card_limit:
            dropped_wide.append(c)
        else:
            cat_cols.append(c)
    else:
        dropped_wide.append(c)

tokens = ["?", "-", "NA", "N/A"]
placeholder_cols, found_tokens = set(), set()
for c in cat_cols:
    st = train[c].astype("string").str.strip()
    for tok in tokens:
        if st.eq(tok).any():
            found_tokens.add(tok)
            placeholder_cols.add(c)
    if st.eq("").any():
        found_tokens.add("")
        placeholder_cols.add(c)

width = len(num_cols) + sum(train[c].nunique(dropna=False) + 1 for c in cat_cols)
wide = width > 300

num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
cat_steps = [(f"ph_{i}", SimpleImputer(missing_values=tok, strategy="most_frequent")) for i, tok in enumerate(sorted(found_tokens))]
cat_steps += [("impute", SimpleImputer(strategy="most_frequent")),
              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=wide))]
cat_pipe = Pipeline(cat_steps)
if not wide:
    num_pipe.set_output(transform="pandas")
    cat_pipe.set_output(transform="pandas")

transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
if not wide:
    pre.set_output(transform="pandas")
pre = mlca_runtime.fit_record(pre, train[feature_cols], "train")
Xt_val = pre.transform(val[feature_cols])

print(f"placeholders: {sorted(found_tokens) if found_tokens else 'none'} in {len(placeholder_cols)} columns")
print(f"features: {len(feature_cols)} num={len(num_cols)} cat={len(cat_cols)} dropped={len(dropped_wide)}")
print(f"derived excluded: {list(derived_reasons) if derived_reasons else 'none'}")
if corr_warnings:
    print(f"corr>=0.98 kept: {sorted(set(corr_warnings))}")
print(f"train {train.shape} val {val.shape} -> {Xt_val.shape}")
mlca_runtime.save_model(pre, "preprocessor.pkl")
