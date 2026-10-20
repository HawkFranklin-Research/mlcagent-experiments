import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["bank_marketing_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

def need(key):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {list(protocol)}")
    return protocol[key]

targets = list(need("targets"))
excluded = list(need("excluded_columns"))
task = need("task")
group_col = protocol.get("group_column")

drop_cols = set(targets) | set(excluded)
if group_col is not None:
    drop_cols.add(group_col)

candidates = [c for c in df.columns if c not in drop_cols]
n_rows = len(df)
features = []
for c in candidates:
    s = df[c]
    is_text = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
    is_int = pd.api.types.is_integer_dtype(s)
    if n_rows > 0 and s.nunique(dropna=False) == n_rows and (is_text or is_int):
        continue
    features.append(c)

numeric_cols = []
categorical_cols = []
for c in features:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

text_cols = [c for c in features if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
placeholder_tokens = {"?", "-", "NA", "N/A", ""}
found_tokens = set()
placeholder_cols = set()
for c in text_cols:
    s = df[c].dropna().astype(str).str.strip()
    for token in placeholder_tokens:
        if (s == token).any():
            found_tokens.add("<blank>" if token == "" else token)
            placeholder_cols.add(c)
print(f"Placeholders found: {sorted(found_tokens)} in {len(placeholder_cols)} columns")

target_col = targets[0] if len(targets) == 1 and targets[0] in df.columns else None
classes = sorted(df[target_col].dropna().unique().tolist()) if target_col else []
groups = [("All", df)]
if target_col and len(features) * (1 + len(classes)) <= 50:
    for cls in classes:
        groups.append((f"{target_col}={cls}", df[df[target_col] == cls]))

rows = []
for gname, gdf in groups:
    for c in numeric_cols:
        s = pd.to_numeric(gdf[c], errors="coerce")
        rows.append({
            "variable": c, "type": "numeric", "group": gname,
            "n": int(s.notna().sum()), "n_missing": int(s.isna().sum()),
            "mean": s.mean(), "sd": s.std(), "min": s.min(),
            "median": s.median(), "max": s.max(),
            "n_levels": np.nan, "top_share": np.nan,
        })
    for c in categorical_cols:
        s = gdf[c]
        vc = s.value_counts(dropna=True)
        rows.append({
            "variable": c, "type": "categorical", "group": gname,
            "n": int(s.notna().sum()), "n_missing": int(s.isna().sum()),
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
            "n_levels": int(s.nunique(dropna=True)),
            "top_share": float(vc.iloc[0] / vc.sum()) if len(vc) else np.nan,
        })

dist = pd.DataFrame(rows)
if len(dist) > 50:
    overall = dist[dist["group"] == "All"]
    dist = overall if len(overall) <= 50 else dist.head(50)

for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    dist[col] = dist[col].map(lambda v: float(f"{v:.4g}") if pd.notna(v) else np.nan)

mlca_runtime.save_table(dist, "distributions.csv", caption="Distribution summaries by feature and target class")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(categorical_cols), source="distributions.csv")
print(f"rows={len(df)} features={len(features)} numeric={len(numeric_cols)} categorical={len(categorical_cols)}")
print(f"distributions rows={len(dist)}")
