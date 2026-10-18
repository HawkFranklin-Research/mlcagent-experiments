import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["optical_digits_uci"])


def need(d, k):
    if k not in d:
        raise KeyError(f"missing key {k!r}; available top-level keys: {sorted(d.keys())}")
    return d[k]


targets = list(need(protocol, "targets"))
excluded = list(need(protocol, "excluded_columns"))
group = protocol.get("group_column", None)

drop = set(targets) | set(excluded)
if group:
    drop.add(group)


def is_keylike(s):
    s2 = s.dropna()
    if s2.shape[0] <= 1:
        return False
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        return s2.nunique() == s2.shape[0]
    if pd.api.types.is_integer_dtype(s):
        return s2.nunique() == s2.shape[0]
    return False


features = [c for c in df.columns if c not in drop and not is_keylike(df[c])]

rows = []
for c in features:
    s = df[c]
    n_missing = int(s.isna().sum())
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        rows.append({
            "feature": c, "type": "numeric", "n_missing": n_missing,
            "mean": s.mean(), "sd": s.std(), "min": s.min(),
            "median": s.median(), "max": s.max(),
            "n_levels": np.nan, "top_level_share": np.nan,
        })
    else:
        vc = s.dropna().value_counts(normalize=True)
        rows.append({
            "feature": c, "type": "categorical", "n_missing": n_missing,
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
            "n_levels": float(s.nunique(dropna=True)),
            "top_level_share": float(vc.iloc[0]) if len(vc) else np.nan,
        })

summary = pd.DataFrame(rows)
for col in ["mean", "sd", "min", "median", "max", "n_levels", "top_level_share"]:
    summary[col] = summary[col].map(lambda v: np.nan if pd.isna(v) else float(f"{v:.4g}"))

n_num = int((summary["type"] == "numeric").sum())
n_cat = int((summary["type"] == "categorical").sum())

mlca_runtime.save_table(summary, "distributions.csv", caption="Per-feature distribution summary")
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")

print(f"rows={df.shape[0]} features={len(features)}")
print(f"n_numeric={n_num} n_categorical={n_cat}")
print(f"summary_shape={summary.shape}")
