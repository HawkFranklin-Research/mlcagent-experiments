import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(key):
    if key not in protocol:
        raise KeyError(f"_protocol missing '{key}'; available top-level keys: {sorted(protocol)}")
    return protocol[key]

targets = list(need("targets"))
excluded = list(need("excluded_columns"))
need("task")
group = protocol.get("group_column")
train = pd.read_parquet(inputs["train"])
n = len(train)

reserved = set(targets) | set(excluded) | ({group} if group else set())
feats, keylike = [], []
for c in [c for c in train.columns if c not in reserved]:
    s = train[c]
    if not pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) == n:
        keylike.append(c); continue
    if pd.api.types.is_integer_dtype(s):
        u = np.unique(s.dropna().to_numpy())
        if len(u) == n and n > 2 and np.all(np.diff(u) == np.diff(u)[0]):
            keylike.append(c); continue
    feats.append(c)

num_feats = [c for c in feats if pd.api.types.is_numeric_dtype(train[c])]
derived = {}
for t in targets:
    yv = train[t].to_numpy(dtype=float)
    for c in num_feats:
        if np.allclose(train[c].to_numpy(dtype=float), yv):
            derived[c] = f"identical to {t}"
    base = [c for c in num_feats if c not in derived and np.nanmin(train[c].to_numpy(dtype=float)) >= 0]
    for i in range(len(base)):
        vi = train[base[i]].to_numpy(dtype=float)
        for j in range(i + 1, len(base)):
            s2 = vi + train[base[j]].to_numpy(dtype=float)
            if np.allclose(s2, yv):
                for c in (base[i], base[j]):
                    derived.setdefault(c, f"row sum {base[i]}+{base[j]} equals {t}")
            for k in range(j + 1, len(base)):
                if np.allclose(s2 + train[base[k]].to_numpy(dtype=float), yv):
                    for c in (base[i], base[j], base[k]):
                        derived.setdefault(c, f"row sum {base[i]}+{base[j]}+{base[k]} equals {t}")

feats = [c for c in feats if c not in derived]
num_feats = [c for c in num_feats if c not in derived]

warn = set()
for t in targets:
    if not pd.api.types.is_numeric_dtype(train[t]):
        continue
    yv = train[t].to_numpy(dtype=float)
    for c in num_feats:
        xv = train[c].to_numpy(dtype=float)
        ok = ~(np.isnan(xv) | np.isnan(yv))
        if ok.sum() > 2 and abs(np.corrcoef(xv[ok], yv[ok])[0, 1]) >= 0.98:
            warn.add(c)

tokens = ["?", "-", "NA", "N/A"]
ph, text_cols = {}, [c for c in feats if not pd.api.types.is_numeric_dtype(train[c])]
for c in text_cols:
    s = train[c].astype("string").str.strip()
    hits = [t for t in tokens if (s == t).any()]
    if hits:
        ph[c] = hits

hi_card = [c for c in text_cols if train[c].nunique(dropna=True) > 0.05 * n]
feats = [c for c in feats if c not in hi_card]
num_cols = [c for c in feats if c in num_feats]
cat_cols = [c for c in feats if c not in num_cols]

est_width = len(num_cols) + sum(int(train[c].nunique(dropna=True)) + 1 for c in cat_cols)
num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat_blocks = []
for c in cat_cols:
    steps = [(f"ph{i}", SimpleImputer(missing_values=t, strategy="most_frequent")) for i, t in enumerate(ph.get(c, []))]
    steps += [("imp", SimpleImputer(strategy="most_frequent")),
              ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    cat_blocks.append((f"cat_{c}", Pipeline(steps), [c]))

blocks = ([("num", num_pipe, num_cols)] if num_cols else []) + cat_blocks
pre = ColumnTransformer(blocks, remainder="drop", sparse_threshold=1.0 if est_width > 300 else 0.3)
if est_width <= 300:
    pre.set_output(transform="pandas")
    num_pipe.set_output(transform="pandas")
    for _, pipe, _ in cat_blocks:
        pipe.set_output(transform="pandas")

X = train[feats]
fitted = mlca_runtime.fit_record(pre, X, "train", y=train[targets])
mlca_runtime.save_model(fitted, "preprocessor.pkl")

print(f"train {train.shape}; features used {len(feats)} ({len(num_cols)} numeric, {len(cat_cols)} categorical); key-like dropped {len(keylike)}")
if derived:
    print("excluded derived: " + "; ".join(f"{k} ({v})" for k, v in sorted(derived.items())))
if hi_card:
    print("dropped high-cardinality text: " + ", ".join(sorted(hi_card)))
if warn:
    print("WARNING corr>=0.98 with a target, kept: " + ", ".join(sorted(warn)))
print(f"placeholder tokens found in {len(ph)} text column(s); encoded width ~{est_width}")
