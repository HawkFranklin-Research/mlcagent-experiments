import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

target_list = proto["targets"]
excluded = list(proto.get("excluded_columns", []))
group_col = proto.get("group_column")
group_cols = [group_col] if group_col else []
target = target_list[0]

df = pd.read_parquet(inputs["bank_marketing_uci"])
feature_cols = [c for c in df.columns if c not in target_list and c not in excluded and c not in group_cols]

num_cols, cat_cols = [], []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
        num_cols.append(c)
    else:
        cat_cols.append(c)

ph_tokens = {"?", "-", "NA", "N/A", "na", "n/a", ""}
ph_cols = []
for c in cat_cols:
    s = df[c].dropna().astype(str)
    stripped = s.str.strip()
    if stripped.isin(ph_tokens).any() or (s != stripped).any():
        ph_cols.append(c)
print(f"placeholders: {sorted(ph_tokens)} in {len(ph_cols)} columns {ph_cols}")

rows = []
group_specs = [("all", df)]
if target in df.columns:
    for v in sorted(df[target].dropna().unique().tolist()):
        group_specs.append((f"{target}={v}", df[df[target] == v]))

for gname, gdf in group_specs:
    for c in num_cols:
        s = pd.to_numeric(gdf[c], errors="coerce")
        rows.append(dict(variable=c, kind="numeric", group=gname,
                         mean=s.mean(), sd=s.std(), min=s.min(), median=s.median(),
                         max=s.max(), n_missing=int(s.isna().sum()),
                         n_levels=np.nan, top_share=np.nan))
    for c in cat_cols:
        s = gdf[c].dropna().astype(str).str.strip()
        n = len(s)
        if n > 0:
            vc = s.value_counts()
            nl, ts = int(len(vc)), float(vc.iloc[0]) / n
        else:
            nl, ts = 0, np.nan
        rows.append(dict(variable=c, kind="categorical", group=gname,
                         mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                         n_missing=int(gdf[c].isna().sum()), n_levels=nl, top_share=ts))

out = pd.DataFrame(rows).round(4)
mlca_runtime.save_table(out, "distributions.csv",
                        caption="Distributions summary per variable and target group")
mlca_runtime.record_fact_candidate("n_numeric", len(num_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")
print(f"rows={len(out)} num={len(num_cols)} cat={len(cat_cols)} groups={len(group_specs)}")
