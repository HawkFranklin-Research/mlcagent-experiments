import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in proto:
        raise KeyError(f"missing protocol key {k!r}; available keys: {sorted(proto)}")
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group = proto.get("group_column")
train = pd.read_parquet(paths["train"])
n = len(train)

drop = set(targets) | set(excluded) | ({group} if group else set())
cand = [c for c in train.columns if c not in drop]

keylike = []
for c in cand:
    s = train[c].dropna()
    if len(s) < 2 or s.nunique() != len(s):
        continue
    if s.dtype == object or pd.api.types.is_integer_dtype(s):
        keylike.append(c)
feats = [c for c in cand if c not in keylike]

placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols = [c for c in feats if train[c].dtype == object and
           train[c].astype(str).str.strip().isin(placeholders).any()]
dates = [c for c in feats if pd.api.types.is_datetime64_any_dtype(train[c])]
for c in feats:
    if c not in dates and train[c].dtype == object:
        parsed = pd.to_datetime(train[c], errors="coerce", format="mixed")
        if parsed.notna().mean() > 0.9:
            dates.append(c)
feats = [c for c in feats if c not in dates]

num, cat, wide = [], [], []
for c in feats:
    s = train[c]
    if s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype):
        k = s.nunique(dropna=True)
        (wide if k > 0.05 * n else cat).append(c)
    elif pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        num.append(c)
    else:
        wide.append(c)

skewed = [c for c in num if train[c].min() >= 0 and
          abs(float(train[c].skew() or 0.0)) > 1.0]

parts = []
for name, cols, use_log in (("num", [c for c in num if c not in skewed], False),
                            ("lognum", skewed, True)):
    if not cols:
        continue
    steps = [("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))] if use_log else []
    steps += [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    parts.append((name, Pipeline(steps), cols))
if cat:
    parts.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                   ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                            sparse_output=False))]), cat))

width = len(num) + sum(int(train[c].nunique(dropna=True)) for c in cat)
pre = ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)
if width <= 300:
    pre.set_output(transform="pandas")

mlca_runtime.fit_record(pre, train[feats], "train")
mlca_runtime.save_model(pre, "preprocessor.pkl")
print(f"features={len(feats)} numeric={len(num)} categorical={len(cat)} dropped_wide={len(wide)}")
print(f"log1p={skewed} dates_dropped={dates} keylike={keylike}")
print(f"placeholder_tokens_columns={len(ph_cols)} targets={targets} excluded={excluded}")
print(f"encoded_width={width} set_output_pandas={width <= 300} rows={n}")
