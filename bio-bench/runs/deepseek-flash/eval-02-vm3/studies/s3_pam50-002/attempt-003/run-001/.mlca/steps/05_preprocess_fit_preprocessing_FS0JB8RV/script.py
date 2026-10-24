import mlca_runtime
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
protocol = json.loads(inp["_protocol"].read_text())

for k in ("targets", "excluded_columns", "task"):
    if k not in protocol:
        raise KeyError(f"Missing {k}; available: {list(protocol)}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
primary_metric = protocol["task"]["primary_metric"]

target = targets[0]
drop_cols = set(targets) | set(excluded)
if group_col:
    drop_cols.add(group_col)
candidate = [c for c in train.columns if c not in drop_cols]

placeholders = {"?", "-", "NA", "N/A", ""}
found = {}
for c in candidate:
    s = train[c]
    if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
        m = s.astype("string").str.strip().isin(placeholders) & s.notna()
        n = int(m.sum())
        if n:
            found[c] = n
for df in (train, val):
    for c in candidate:
        s = df[c]
        if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
            z = s.astype("string").str.strip()
            df[c] = z.mask(z.isin(placeholders))
print(f"Placeholders: {len(found)} columns; examples={list(found)[:5]}")

drops = []
features = []
for c in candidate:
    s = train[c]
    if (pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)) and \
       s.nunique(dropna=False) == len(train) and not any(t in c.lower() for t in ("date", "time")):
        drops.append((c, "key-like")); continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(train) and \
       any(t in c.lower() for t in ("id", "code")):
        drops.append((c, "key-like")); continue
    features.append(c)

date_suffix = ["_year", "_month", "_day", "_dow", "_hour"]
for c in list(features):
    s = train[c]
    parsed = None
    if pd.api.types.is_datetime64_any_dtype(s):
        parsed = pd.to_datetime(train[c], errors="coerce")
    elif (pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)) and \
         any(t in c.lower() for t in ("date", "time", "_dt")):
        p = pd.to_datetime(s, errors="coerce")
        if p.notna().mean() > 0.8:
            parsed = p
    if parsed is not None:
        for df in (train, val):
            p = pd.to_datetime(df[c], errors="coerce")
            df[c + "_year"] = p.dt.year
            df[c + "_month"] = p.dt.month
            df[c + "_day"] = p.dt.day
            df[c + "_dow"] = p.dt.dayofweek
            df[c + "_hour"] = p.dt.hour
        features.remove(c)
        features.extend([c + s for s in date_suffix])
        drops.append((c, "date->numeric"))

for c in list(features):
    s = train[c]
    if (pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)) and \
       s.nunique(dropna=False) > 0.05 * len(train):
        features.remove(c); drops.append((c, "high-cardinality"))

y = train[target]
y_s = y.astype("string").fillna("__NA__")
for c in list(features):
    if train[c].astype("string").fillna("__NA__").equals(y_s):
        features.remove(c); drops.append((c, "target derivation"))
if pd.api.types.is_numeric_dtype(y):
    nonneg = [c for c in features if pd.api.types.is_numeric_dtype(train[c]) and train[c].min() >= 0]
    if 2 <= len(nonneg) <= 50:
        for i in range(len(nonneg)):
            for j in range(i + 1, len(nonneg)):
                if (train[nonneg[i]] + train[nonneg[j]]).equals(y.astype(float)):
                    features.remove(nonneg[i]); features.remove(nonneg[j])
                    drops.append((nonneg[i], "target sum")); drops.append((nonneg[j], "target sum")); break
            else: continue
            break
    numf = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    corr = train[numf].corrwith(y.astype(float)).abs()
    warn = corr[corr >= 0.98].index.tolist()
    if warn:
        print(f"Correlation warning: {warn[:5]}")
print(f"Dropped: {drops[:5]}")

num_cols = [c for c in features if pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_bool_dtype(train[c])]
cat_cols = [c for c in features if c not in num_cols]
width = len(num_cols) + sum(max(1, train[c].nunique(dropna=False)) for c in cat_cols)
wide = width > 300
num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=wide))])
preprocessor = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")
if not wide:
    preprocessor.set_output(transform="pandas")
X_train = train[features]
X_val = val[features]
preprocessor = mlca_runtime.fit_record(preprocessor, X_train, "train", y=y)
Xt = preprocessor.transform(X_val)
print(f"Features: {len(features)} ({len(num_cols)} numeric, {len(cat_cols)} categorical); wide={wide}")
print(f"Train {X_train.shape}; val transformed {Xt.shape}")
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
