import mlca_runtime, json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths["_protocol"].read_text())
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group_col = proto.get("group_column")
df = pd.read_parquet(paths["heart_disease_cleveland_uci"])

feat_cols = []
for c in df.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    s = df[c].dropna()
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        if len(s) > 0 and s.nunique() == len(s):
            continue
    elif pd.api.types.is_integer_dtype(s):
        if len(s) > 0 and s.nunique() == len(s):
            continue
    feat_cols.append(c)

ph_tokens = {"?", "-", "NA", "N/A", "na", "n/a", ""}
ph_cols = 0
for c in df.columns:
    if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
        vals = set(df[c].dropna().astype(str).str.strip().unique())
        if vals & ph_tokens:
            ph_cols += 1
print(f"placeholder columns: {ph_cols}")

tgt = targets[0]
groups = [("all", df)] + [(f"{tgt}={v}", df[df[tgt] == v]) for v in sorted(df[tgt].dropna().unique())]

rows = []
for gname, gdf in groups:
    for c in feat_cols:
        s = gdf[c]
        row = {"variable": c, "group": gname, "n": int(s.notna().sum()),
               "n_missing": int(s.isna().sum())}
        if pd.api.types.is_numeric_dtype(s):
            row.update({"kind": "numeric", "mean": s.mean(), "sd": s.std(),
                        "min": s.min(), "median": s.median(), "max": s.max(),
                        "n_levels": np.nan, "top_share": np.nan})
        else:
            vc = s.value_counts(normalize=True, dropna=True)
            row.update({"kind": "categorical", "mean": np.nan, "sd": np.nan,
                        "min": np.nan, "median": np.nan, "max": np.nan,
                        "n_levels": int(s.nunique()),
                        "top_share": float(vc.iloc[0]) if len(vc) else np.nan})
        rows.append(row)

cols = ["variable", "group", "kind", "n", "n_missing", "mean", "sd",
        "min", "median", "max", "n_levels", "top_share"]
out = pd.DataFrame(rows)[cols].round(4)
out = out.head(50)
mlca_runtime.save_table(out, "distributions.csv", "Feature distribution summaries overall and by target class")

base = out[out["group"] == "all"]
n_num = int((base["kind"] == "numeric").sum())
n_cat = int((base["kind"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")
print(f"rows={len(out)} numeric={n_num} categorical={n_cat}")
