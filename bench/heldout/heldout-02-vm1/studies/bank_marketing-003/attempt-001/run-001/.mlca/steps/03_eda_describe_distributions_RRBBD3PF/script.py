import json
import mlca_runtime
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
for key in ["targets", "excluded_columns", "task"]:
    if key not in proto:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {sorted(proto)}")

targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group_col = proto.get("group_column")
if not targets:
    raise ValueError("No targets declared in _protocol")
target = targets[0]

df = pd.read_parquet(inputs["bank_marketing_uci"])
if target not in df.columns:
    raise KeyError(f"Target column {target!r} not found; columns available: {list(df.columns)}")

text_cols = [
    c for c in df.columns
    if c not in targets and c not in excluded and c != group_col
    and (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]))
]
placeholder_tokens = {"?", "-", "NA", "N/A"}
work = df.copy()
ph_cols = 0
ph_count = 0
for c in text_cols:
    s = work[c].astype("string").str.strip()
    mask = (s.isin(placeholder_tokens) | s.eq("")).fillna(False)
    n = int(mask.sum())
    if n:
        ph_cols += 1
        ph_count += n
        work[c] = s.mask(mask, pd.NA)
    else:
        work[c] = s

feature_cols = [
    c for c in df.columns
    if c not in targets and c not in excluded and c != group_col
]

def is_key_like(s: pd.Series) -> bool:
    nn = int(s.notna().sum())
    if nn == 0 or s.nunique(dropna=True) != nn:
        return False
    if pd.api.types.is_integer_dtype(s.dtype):
        return True
    return not (pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype))

feature_cols = [c for c in feature_cols if not is_key_like(work[c])]

def feature_type(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s.dtype):
        return "numeric" if set(s.dropna().unique()).issubset({0, 1, True, False}) else "categorical"
    return "numeric" if pd.api.types.is_numeric_dtype(s.dtype) else "categorical"

types = {c: feature_type(work[c]) for c in feature_cols}
num_cols = [c for c in feature_cols if types[c] == "numeric"]
cat_cols = [c for c in feature_cols if types[c] == "categorical"]

def sig4(x):
    if pd.isna(x):
        return np.nan
    return float(f"{float(x):.4g}")

rows = []

def add_row(c: str, group: str, sub: pd.DataFrame) -> None:
    s = sub[c]
    if types[c] == "numeric":
        rows.append({
            "variable": c, "type": "numeric", "group": group,
            "n": int(s.notna().sum()), "n_missing": int(s.isna().sum()),
            "mean": sig4(s.mean()), "sd": sig4(s.std()), "min": sig4(s.min()),
            "median": sig4(s.median()), "max": sig4(s.max()),
            "n_levels": np.nan, "top_level_share": np.nan,
        })
    else:
        shares = s.value_counts(normalize=True, dropna=True)
        rows.append({
            "variable": c, "type": "categorical", "group": group,
            "n": int(s.notna().sum()), "n_missing": int(s.isna().sum()),
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
            "n_levels": int(s.nunique(dropna=True)),
            "top_level_share": sig4(shares.iloc[0]) if len(shares) else np.nan,
        })

for c in feature_cols:
    add_row(c, "all", work)

classes = list(pd.unique(work[target].dropna()))
for cls in classes:
    sub = work[work[target] == cls]
    for c in feature_cols:
        if len(rows) >= 50:
            break
        add_row(c, str(cls), sub)
    if len(rows) >= 50:
        break

summary = pd.DataFrame(rows)
if len(summary) > 50:
    summary = summary.head(50).copy()

n_numeric = int(summary.loc[summary["type"] == "numeric", "variable"].nunique())
n_categorical = int(summary.loc[summary["type"] == "categorical", "variable"].nunique())
mlca_runtime.save_table(summary, "distributions.csv", caption="Feature distributions by target class")
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")
print(f"features: {len(feature_cols)} ({len(num_cols)} numeric, {len(cat_cols)} categorical)")
print(f"summary rows: {len(summary)}")
print(f"placeholder tokens in {ph_cols} text columns: {ph_count}")
