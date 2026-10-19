import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["dry_bean_uci"])
protocol = json.loads(paths["_protocol"].read_text())

for key in ("targets", "excluded_columns", "group_column"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(protocol.keys())}")

targets = protocol["targets"]
if isinstance(targets, str):
    targets = [targets]
excluded = set(protocol["excluded_columns"])
group_col = protocol["group_column"]

def is_key_like(s, name):
    if s.isna().any() or s.nunique(dropna=True) != len(s):
        return False
    lname = str(name).lower()
    if any(tok in lname for tok in ("id", "index", "key", "code")):
        return True
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.is_monotonic_increasing
    return False

excluded_cols = set(targets) | excluded
if group_col and group_col in df.columns:
    excluded_cols.add(group_col)
candidate_cols = [c for c in df.columns if c not in excluded_cols]
feature_cols = [c for c in candidate_cols if not is_key_like(df[c], c)]

num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]

text_feature_cols = [
    c for c in feature_cols
    if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)
]
placeholders = {"?", "-", "NA", "N/A"}
found = set()
n_cols_with_ph = 0
for c in text_feature_cols:
    s = df[c].astype("string").str.strip()
    mask = s.isin(list(placeholders) + [""])
    if mask.any():
        n_cols_with_ph += 1
        found.update(s[mask].dropna().unique().tolist())

rows = []
for c in num_cols:
    s = df[c]
    rows.append({
        "variable": c, "type": "numeric", "target_class": np.nan,
        "mean": s.mean(), "sd": s.std(), "min": s.min(), "median": s.median(),
        "max": s.max(), "n_missing": int(s.isna().sum()),
        "n_levels": np.nan, "top_level_share": np.nan,
    })

for c in cat_cols:
    s = df[c]
    vc = s.value_counts(dropna=True, normalize=True)
    rows.append({
        "variable": c, "type": "categorical", "target_class": np.nan,
        "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan,
        "max": np.nan, "n_missing": np.nan,
        "n_levels": int(s.nunique(dropna=True)),
        "top_level_share": float(vc.iloc[0]) if len(vc) else np.nan,
    })

remaining = 50 - len(rows)
if remaining > 0:
    for t in targets:
        if t not in df.columns:
            continue
        nlev = int(df[t].nunique(dropna=True))
        for cls, cnt in df[t].value_counts(dropna=True).items():
            if remaining <= 0:
                break
            rows.append({
                "variable": t, "type": "target_class", "target_class": cls,
                "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan,
                "max": np.nan, "n_missing": np.nan,
                "n_levels": nlev, "top_level_share": float(cnt / len(df)),
            })
            remaining -= 1
        if remaining <= 0:
            break

summary = pd.DataFrame(rows, columns=[
    "variable", "type", "target_class", "mean", "sd", "min", "median",
    "max", "n_missing", "n_levels", "top_level_share",
])

def round_sig(v, sig=4):
    if pd.isna(v):
        return v
    v = float(v)
    if v == 0:
        return 0.0
    return round(v, sig - int(np.floor(np.log10(abs(v)))) - 1)

for c in ("mean", "sd", "min", "median", "max", "n_missing", "n_levels", "top_level_share"):
    summary[c] = summary[c].map(round_sig)

mlca_runtime.save_table(
    summary,
    "distributions.csv",
    caption="Distribution summaries for numeric and categorical variables, with target-class rows where useful",
)

mlca_runtime.record_fact_candidate("n_numeric", len(num_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")

print(f"rows={len(df)} features={len(feature_cols)}")
print(f"numeric={len(num_cols)} categorical={len(cat_cols)}")
print(f"distributions rows={len(summary)}")
print(f"placeholders={sorted(found)} in {n_cols_with_ph} text columns")
