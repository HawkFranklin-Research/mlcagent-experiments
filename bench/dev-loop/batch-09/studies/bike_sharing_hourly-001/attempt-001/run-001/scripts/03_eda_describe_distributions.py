import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())

available = sorted(proto.keys())
for key in ("targets", "excluded_columns", "task"):
    if key not in proto:
        raise KeyError(f"Protocol missing {key!r}; available top-level keys: {available}")

targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
task = proto["task"]
group = proto.get("group_column")

df = pd.read_parquet(inp["bike_sharing_hourly_uci"])

drop = set(targets) | set(excluded)
if group:
    drop.add(group)

feature_cols = []
for c in df.columns:
    if c in drop:
        continue
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        non = s.dropna().astype(str).str.strip()
        if len(non) > 0 and non.nunique() == len(non):
            continue
    feature_cols.append(c)

placeholders = {"?", "-", "NA", "N/A"}
ph_found = set()
ph_cols = 0
for c in df.select_dtypes(include=["object", "string", "category"]).columns:
    raw = df[c]
    stripped = raw.dropna().astype(str).str.strip()
    hits = set(stripped[stripped.isin(placeholders)].unique())
    if (raw.notna() & raw.astype(str).str.strip().eq("")).any():
        hits.add("<whitespace>")
    if hits:
        ph_cols += 1
        ph_found |= hits
if ph_cols:
    print(f"Placeholders found: {sorted(ph_found)} in {ph_cols} columns")
else:
    print("Placeholders found: none")

rows = []
for c in df.columns:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    n_missing = int(s.isna().sum())
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        vals = pd.to_numeric(s, errors="coerce")
        rows.append({
            "column": c, "type": "numeric", "n_missing": n_missing,
            "mean": vals.mean(), "sd": vals.std(), "min": vals.min(),
            "median": vals.median(), "max": vals.max(),
            "n_levels": np.nan, "top_level_share": np.nan,
        })
    else:
        non = s.dropna()
        vc = non.astype(str).str.strip().value_counts() if len(non) else pd.Series(dtype=int)
        rows.append({
            "column": c, "type": "categorical", "n_missing": n_missing,
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
            "n_levels": int(non.nunique(dropna=True)),
            "top_level_share": (vc.iloc[0] / len(non)) if len(non) else np.nan,
        })

summary = pd.DataFrame(rows).head(50)
for c in ["mean", "sd", "min", "median", "max", "top_level_share"]:
    summary[c] = pd.to_numeric(summary[c], errors="coerce").map(
        lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan
    )
summary["n_missing"] = summary["n_missing"].astype(int)
summary["n_levels"] = summary["n_levels"].map(lambda x: int(x) if pd.notna(x) else np.nan)

mlca_runtime.save_table(summary, "distributions.csv", caption="Column distribution summary")
n_numeric = int((summary["type"] == "numeric").sum())
n_categorical = int((summary["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")
print(f"distributions.csv: {len(summary)} rows, {n_numeric} numeric, {n_categorical} categorical")
