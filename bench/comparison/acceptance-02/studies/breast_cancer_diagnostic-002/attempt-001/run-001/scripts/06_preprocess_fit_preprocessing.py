import json
from itertools import combinations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
protocol = json.loads(inp["_protocol"].read_text())

need = {"targets", "excluded_columns", "task"}
if not need <= set(protocol):
    raise KeyError(f"protocol missing {need - set(protocol)}; available {list(protocol)}")
targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")

n = len(train)
key_like = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == n:
        key_like.append(c)
    elif pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == n and s.min() == 0 and s.max() == n - 1:
        key_like.append(c)

features = [c for c in train.columns if c not in set(targets) | excluded | set(key_like) and c != group]
derived = set()
for t in targets:
    if t not in train:
        continue
    for c in features:
        if train[c].astype("string").fillna("__NA__").equals(train[t].astype("string").fillna("__NA__")):
            derived.add(c)
    if pd.api.types.is_numeric_dtype(train[t]):
        tn = pd.to_numeric(train[t], errors="coerce")
        nums = [c for c in features if c not in derived and pd.api.types.is_numeric_dtype(train[c])
                and train[c].notna().all() and (train[c] >= 0).all()]
        if tn.notna().all():
            for r in (1, 2, 3):
                for combo in combinations(nums, r):
                    if (train[list(combo)].sum(axis=1) == tn).all():
                        derived.update(combo)
derived = sorted(derived)
features = [c for c in features if c not in set(derived)]
print("excluded:", [(c, "key-like") for c in key_like] + [(c, "derivation") for c in derived])

corr_warn = set()
for t in targets:
    if t in train and pd.api.types.is_numeric_dtype(train[t]):
        yt = pd.to_numeric(train[t], errors="coerce")
        for c in features:
            if pd.api.types.is_numeric_dtype(train[c]):
                if abs(pd.to_numeric(train[c], errors="coerce").corr(yt)) >= 0.98:
                    corr_warn.add(c)
if corr_warn:
    print("corr>=0.98 kept:", sorted(corr_warn))

Xtr = train[features].copy()
Xva = val[features].copy()
placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols = [c for c in features if not pd.api.types.is_numeric_dtype(Xtr[c])]
found, ph_count = set(), 0
for c in ph_cols:
    Xtr[c] = Xtr[c].astype("string")
    Xva[c] = Xva[c].astype("string")
    s = Xtr[c].str.strip()
    m = (s.isin(placeholders) | s.eq("")).fillna(False)
    if m.any():
        ph_count += 1
        found.update(s[m].dropna().unique().tolist())
        Xtr.loc[m, c] = pd.NA
        sv = Xva[c].str.strip()
        Xva.loc[(sv.isin(placeholders) | sv.eq("")).fillna(False), c] = pd.NA
print("placeholder_columns:", ph_count, "tokens:", sorted(found))

for c in list(features):
    if pd.api.types.is_datetime64_any_dtype(Xtr[c]):
        Xtr[c + "_dayofweek"] = Xtr[c].dt.dayofweek
        Xva[c + "_dayofweek"] = Xva[c].dt.dayofweek
        features.remove(c)
        features.append(c + "_dayofweek")

cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(Xtr[c])]
num_cols = [c for c in features if c not in cat_cols]
limit = 0.05 * len(Xtr)
high = [c for c in cat_cols if Xtr[c].nunique(dropna=True) > limit]
if high:
    print("dropped_high_cardinality:", high)
    cat_cols = [c for c in cat_cols if c not in high]
    features = [c for c in features if c not in high]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
]).set_output(transform="pandas")
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
]).set_output(transform="pandas")

transformers = []
if num_cols:
    transformers.append(("num", num_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", cat_pipe, cat_cols))
pre = ColumnTransformer(transformers, remainder="drop").set_output(transform="pandas")

cols = num_cols + cat_cols
pre = mlca_runtime.fit_record(pre, Xtr[cols], "train")
Xt_val = pre.transform(Xva[cols])
mlca_runtime.save_model(pre, "preprocessor.pkl")
print(f"features={len(cols)} numeric={len(num_cols)} categorical={len(cat_cols)} val={Xt_val.shape}")
