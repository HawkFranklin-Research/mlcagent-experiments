import mlca_runtime, json
import numpy as np, pandas as pd
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
protocol = json.loads(inp["_protocol"].read_text())
try:
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available: {list(protocol)}")
group = protocol.get("group_column")
base_ex = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in base_ex]
n = len(train)
key_like = []
for c in list(feature_cols):
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if n > 1 and s.nunique(dropna=True) == n:
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s):
        u = np.sort(s.dropna().unique())
        if len(u) == n and n > 1 and (len(u) < 3 or np.all(np.diff(u) == np.diff(u)[0])):
            key_like.append(c)
for c in key_like:
    feature_cols.remove(c)

derived = set()
for t in targets:
    y = train[t]
    for c in feature_cols:
        if (train[c].astype(str) == y.astype(str)).all():
            derived.add(c)
    if pd.api.types.is_numeric_dtype(y):
        num_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) and (train[c].dropna() >= 0).all()]
        for k in (2, 3):
            for combo in combinations(num_feats, k):
                if set(combo) & derived:
                    continue
                if np.allclose(train[list(combo)].sum(axis=1).values, y.values, rtol=0, atol=0, equal_nan=False):
                    derived.update(combo)
for c in derived:
    if c in feature_cols:
        feature_cols.remove(c)
if derived:
    print("excluded derivations (target derivation): " + ", ".join(sorted(derived)))

warn = []
for t in targets:
    y = train[t]
    if pd.api.types.is_numeric_dtype(y):
        for c in feature_cols:
            if pd.api.types.is_numeric_dtype(train[c]):
                r = train[c].corr(y)
                if pd.notna(r) and abs(r) >= 0.98:
                    warn.append(c)
if warn:
    print("high corr kept: " + ", ".join(sorted(set(warn))))

for c in [c for c in feature_cols if pd.api.types.is_datetime64_any_dtype(train[c])]:
    for df in (train, val):
        d = pd.to_datetime(df[c])
        df[c + "_year"] = d.dt.year; df[c + "_month"] = d.dt.month
        df[c + "_dow"] = d.dt.dayofweek; df[c + "_hour"] = d.dt.hour
    feature_cols.remove(c)
    feature_cols.extend([c + "_year", c + "_month", c + "_dow", c + "_hour"])

text_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]) or isinstance(train[c].dtype, pd.CategoricalDtype)]
tokens = ["?", "-", "NA", "N/A", ""]
found = set(); cols_with = 0
for c in text_cols:
    s = train[c].astype(str).str.strip()
    m = s.isin(tokens)
    if m.any():
        cols_with += 1
        vals = set(s[m].unique())
        if "" in vals:
            vals.remove(""); vals.add("<whitespace>")
        found.update(vals)
print(f"placeholders found: {', '.join(sorted(found)) if found else 'none'}; {cols_with} text columns")

num_cols, cat_cols, drop_high = [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    elif s.nunique(dropna=True) > 0.05 * n:
        drop_high.append(c)
    else:
        cat_cols.append(c)
for c in drop_high:
    feature_cols.remove(c)
for c in cat_cols:
    train[c] = train[c].astype(object).where(train[c].notna(), np.nan)
    val[c] = val[c].astype(object).where(val[c].notna(), np.nan)

est = len(num_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
wide = est > 300

cat_steps = []
for i, tok in enumerate(tokens):
    cat_steps.append((f"ph{i}", SimpleImputer(missing_values=tok, strategy="constant", fill_value=np.nan)))
cat_steps.append(("most_frequent", SimpleImputer(strategy="most_frequent")))
cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=wide)))
cat_pipe = Pipeline(cat_steps)
num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])

trans = []
if num_cols:
    trans.append(("num", num_pipe, num_cols))
if cat_cols:
    trans.append(("cat", cat_pipe, cat_cols))
pre = ColumnTransformer(trans, remainder="drop")
if not wide:
    pre.set_output(transform="pandas")
pre = mlca_runtime.fit_record(pre, train[feature_cols], "train")
Xt = pre.transform(val[feature_cols])
print(f"fitted {len(feature_cols)} features; val width {Xt.shape[1]}")
mlca_runtime.save_model(pre, "preprocessor.pkl")
