import mlca_runtime
import json
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]; excluded = protocol["excluded_columns"]
group = protocol.get("group_column"); task = protocol["task"]
train = pd.read_parquet(inputs["train"]); val = pd.read_parquet(inputs["val"])
n = len(train)

keep_out = set(targets) | set(excluded) | ({group} if group else set())

def keylike(s):
    if s.isna().any() or s.nunique(dropna=False) != n:
        return False
    if isinstance(s.dtype, object) or str(s.dtype).startswith('string'):
        return True
    if pd.api.types.is_integer_dtype(s):
        mn, mx = int(s.min()), int(s.max())
        return (mn == 1 and mx == n) or (mn == 0 and mx == n - 1)
    return False

features = [c for c in train.columns if c not in keep_out and not keylike(train[c])]
y_tr = train[targets[0]]
target_vals = y_tr.astype(float).values
tmax = float(target_vals.max())

def combo_sum(cols):
    out = train[cols[0]].astype(float).values.copy()
    for c in cols[1:]:
        out = out + train[c].astype(float).values
    return out

excluded_feats = []
for c in features:
    if pd.api.types.is_numeric_dtype(train[c]):
        try:
            if np.array_equal(train[c].fillna(-999).astype(float).values, target_vals):
                excluded_feats.append((c, "eq_target"))
        except Exception:
            pass

nn = [c for c in features if c not in [e[0] for e in excluded_feats]
      and pd.api.types.is_numeric_dtype(train[c])
      and (train[c].fillna(-1) >= 0).all()
      and float(train[c].max()) <= tmax]
found = set()
for size in (2, 3):
    for combo in combinations(nn, size):
        s = combo_sum(list(combo))
        if not np.array_equal(s[:200], target_vals[:200]):
            continue
        if np.array_equal(s, target_vals):
            for c in combo:
                found.add(c)
for c in found:
    excluded_feats.append((c, "row_sum_target"))
for c, w in excluded_feats:
    print(f"Excluded {c}: {w}")
exc_names = set(e[0] for e in excluded_feats)
features = [c for c in features if c not in exc_names]

text_cols = [c for c in features if isinstance(train[c].dtype, object)
             or str(train[c].dtype).startswith('string')]
pc = sum(1 for c in text_cols
         if any((train[c].astype(str).str.strip() == t).any() for t in ['?', '-', 'NA', 'N/A', '']))
print(f"Placeholder columns: {pc}/{len(text_cols)} text")

numeric_cols, cat_cols, high_card = [], [], []
for c in features:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        high_card.append(c); continue
    if pd.api.types.is_bool_dtype(s) or (pd.api.types.is_integer_dtype(s)
                                          and set(s.dropna().unique()).issubset({0, 1})):
        numeric_cols.append(c); continue
    if pd.api.types.is_numeric_dtype(s):
        (high_card if s.nunique(dropna=True) > max(20, 0.05 * n) else numeric_cols).append(c)
    else:
        (high_card if s.nunique(dropna=True) > max(20, 0.05 * n) else cat_cols).append(c)
if high_card:
    print(f"Dropped high-card/datetime: {len(high_card)}")

est = len(numeric_cols) + sum(min(train[c].nunique(dropna=True), 200) for c in cat_cols)
use_pandas = est <= 300
pre_num = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
pre_cat = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                    ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=not use_pandas))])
tfs = []
if numeric_cols: tfs.append(('num', pre_num, numeric_cols))
if cat_cols: tfs.append(('cat', pre_cat, cat_cols))
ct = ColumnTransformer(tfs, remainder='drop')
if use_pandas:
    ct.set_output(transform='pandas')
    pre_num.set_output(transform='pandas')
    pre_cat.set_output(transform='pandas')

mlca_runtime.fit_record(ct, train[features], "train", y=y_tr)
Xv = ct.transform(val[features])
print(f"Fit ok; val={getattr(Xv, 'shape', None)}")
mlca_runtime.save_model(ct, "preprocessor.pkl")
