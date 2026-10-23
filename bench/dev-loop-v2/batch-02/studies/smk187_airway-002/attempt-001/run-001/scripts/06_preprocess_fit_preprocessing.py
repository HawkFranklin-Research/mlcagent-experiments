import json
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing protocol key {k!r}; available keys: {sorted(d.keys())}")
    return d[k]

targets = list(need(protocol, "targets"))
excluded = list(need(protocol, "excluded_columns"))
group_col = protocol.get("group_column")
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
y = train[targets[0]]

drop_set = set(targets) | set(excluded)
if group_col is not None:
    drop_set.add(group_col)
cand = [c for c in train.columns if c not in drop_set]

key_like = []
for c in cand:
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == len(s) > 1:
        key_like.append(c)
    elif pd.api.types.is_integer_dtype(s):
        name = c.lower()
        if s.nunique(dropna=True) == len(s) and (name in {"id", "row_id", "index", "sample_id"} or name.endswith("_id")):
            key_like.append(c)
cand = [c for c in cand if c not in key_like]

placeholder_tokens = {"?", "-", "NA", "N/A", ""}
found_tokens, ph_cols = set(), []
for c in cand:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        stripped = s.astype("string").str.strip()
        mask = stripped.isin(placeholder_tokens)
        if mask.any():
            ph_cols.append(c)
            found_tokens |= set(stripped[mask].dropna().unique())
print(f"Placeholder tokens {sorted(found_tokens)} in {len(ph_cols)} text columns")

excluded_derived = []
y_is_num = pd.api.types.is_numeric_dtype(y)
for c in cand:
    s = train[c]
    eq = False
    if y_is_num and pd.api.types.is_numeric_dtype(s):
        eq = np.allclose(s.to_numpy(), y.to_numpy(), rtol=0, atol=1e-12, equal_nan=True)
    elif not y_is_num and not pd.api.types.is_numeric_dtype(s):
        eq = s.astype("string").equals(y.astype("string"))
    if eq:
        excluded_derived.append((c, "equals target"))

# Up-to-three non-negative column sums are only checked for numeric binary targets.
if y_is_num and set(pd.unique(y.dropna())) <= {0, 1}:
    yv = y.to_numpy()
    neg, pos = yv == 0, yv == 1
    sub = []
    for c in cand:
        s = train[c]
        if pd.api.types.is_numeric_dtype(s):
            v = s.to_numpy()
            if np.all(v >= -1e-12) and np.allclose(v[neg], 0, atol=1e-12) and np.all(v[pos] <= 1 + 1e-12):
                sub.append((c, v))
    for i in range(len(sub)):
        c1, v1 = sub[i]
        for j in range(i + 1, len(sub)):
            c2, v2 = sub[j]
            if np.allclose(v1 + v2, yv, atol=1e-12):
                excluded_derived.extend([(c1, f"sum with {c2} equals target"), (c2, f"sum with {c1} equals target")])
                break
        else:
            continue
        break
    if len(sub) <= 300:
        for i in range(len(sub)):
            c1, v1 = sub[i]
            for j in range(i + 1, len(sub)):
                c2, v2 = sub[j]
                for k in range(j + 1, len(sub)):
                    c3, v3 = sub[k]
                    if np.allclose(v1 + v2 + v3, yv, atol=1e-12):
                        excluded_derived.extend([(c1, "sum of three equals target"), (c2, "sum of three equals target"), (c3, "sum of three equals target")])
                        break
                else:
                    continue
                break
            else:
                continue
            break

excluded_names = {c for c, _ in excluded_derived}
feature_cols = [c for c in cand if c not in excluded_names]
if excluded_derived:
    print("Excluded derivations: " + ", ".join(f"{c} ({r})" for c, r in excluded_derived[:5]))
else:
    print("No exact-derivation features excluded")

numeric_cols, cat_cols, dropped = [], [], list(key_like)
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        dropped.append(c)
    elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    elif s.nunique(dropna=True) > max(1, int(0.05 * len(train))):
        dropped.append(c)
    else:
        cat_cols.append(c)

for c in cat_cols:
    train[c] = train[c].astype("string")
    val[c] = val[c].astype("string")

encoded_width = len(numeric_cols) + sum(train[c].nunique(dropna=True) for c in cat_cols)
wide = encoded_width > 300

num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
cat_steps = [("strip", FunctionTransformer(np.char.strip))]
for i, token in enumerate([" ? ", "-", "NA", "N/A", ""]):
    cat_steps.append((f"ph_{i}", SimpleImputer(missing_values=token.strip(), strategy="constant", fill_value=np.nan)))
cat_steps += [
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not wide)),
]
cat_pipe = Pipeline(cat_steps)

pre = ColumnTransformer(
    [("num", num_pipe, numeric_cols), ("cat", cat_pipe, cat_cols)],
    remainder="drop",
)
if not wide:
    pre.set_output(transform="pandas")
    num_pipe.set_output(transform="pandas")
    cat_pipe.set_output(transform="pandas")

pre = mlca_runtime.fit_record(pre, train[feature_cols], "train", y=y)
Xt_val = pre.transform(val[feature_cols])
print(f"features {len(feature_cols)} (numeric {len(numeric_cols)}, categorical {len(cat_cols)}, dropped {len(dropped)})")
print(f"val transformed {Xt_val.shape}")

mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_transformed_columns", float(Xt_val.shape[1]), "val")
mlca_runtime.save_model(pre, "preprocessor.pkl")
