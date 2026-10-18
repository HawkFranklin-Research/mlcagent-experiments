import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["online_retail_uci"])

for k in ("targets", "excluded_columns", "task"):
    if k not in protocol:
        raise KeyError(f"Missing '{k}' in _protocol. Available keys: {sorted(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")

drop = set(targets) | set(excluded)
if group:
    drop.add(group)


def is_key_like(s, n_notnull):
    nunique = s.nunique(dropna=True)
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        if s.dtype.kind in "iu":
            vals = np.sort(s.dropna().unique())
            if len(vals) > 1 and nunique == n_notnull and np.all(np.diff(vals) == 1):
                return True
    if (pd.api.types.is_string_dtype(s) or s.dtype == object) and nunique == n_notnull and n_notnull > 1:
        return True
    return False


feature_cols = []
for c in df.columns:
    if c in drop:
        continue
    n = df[c].notna().sum()
    if n == 0:
        continue
    if is_key_like(df[c], n):
        continue
    feature_cols.append(c)

rows = []
for c in feature_cols:
    s = df[c]
    miss = int(s.isna().sum())
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        nn = s.dropna()
        rows.append({
            "variable": c, "type": "numeric", "n_missing": miss,
            "mean": float(nn.mean()) if len(nn) else np.nan,
            "sd": float(nn.std(ddof=1)) if len(nn) > 1 else np.nan,
            "min": float(nn.min()) if len(nn) else np.nan,
            "median": float(nn.median()) if len(nn) else np.nan,
            "max": float(nn.max()) if len(nn) else np.nan,
            "n_levels": np.nan, "top_level_share": np.nan,
        })
    else:
        vc = s.value_counts(dropna=True)
        nn = int(s.notna().sum())
        rows.append({
            "variable": c, "type": "categorical", "n_missing": miss,
            "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
            "n_levels": int(vc.size),
            "top_level_share": float(vc.iloc[0] / nn) if nn and vc.size else np.nan,
        })

dist = pd.DataFrame(rows)
for col in ["mean", "sd", "min", "median", "max", "top_level_share"]:
    dist[col] = pd.to_numeric(dist[col], errors="coerce").round(4)

mlca_runtime.save_table(dist, "distributions.csv", caption="Feature distribution summary")

n_num = int((dist["type"] == "numeric").sum())
n_cat = int((dist["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")

print(f"table shape: {df.shape}")
print(f"features summarized: {len(feature_cols)}")
print(f"numeric={n_num} categorical={n_cat}")
