import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
df = pd.read_parquet(inp["bike_sharing_hourly_uci"])

placeholder_cols = 0
for c in df.select_dtypes(include=["object", "string"]).columns:
    s = df[c].astype(str).str.strip()
    if s.isin(["?", "-", "NA", "N/A", ""]).any():
        placeholder_cols += 1
print(f"placeholder token columns: {placeholder_cols}")

rows, n_num, n_cat = [], 0, 0
for c in df.columns:
    s = df.iloc[:, df.columns.get_loc(c)]
    n_miss = int(s.isna().sum())
    is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    if is_num:
        n_num += 1
        rows.append({"variable": c, "kind": "numeric",
                     "mean": s.mean(), "sd": s.std(), "min": s.min(),
                     "median": s.median(), "max": s.max(), "n_missing": n_miss,
                     "n_levels": float("nan"), "top_share": float("nan")})
    else:
        n_cat += 1
        vc = s.astype(str).value_counts(dropna=True)
        rows.append({"variable": c, "kind": "categorical",
                     "mean": float("nan"), "sd": float("nan"), "min": float("nan"),
                     "median": float("nan"), "max": float("nan"), "n_missing": n_miss,
                     "n_levels": int(vc.size),
                     "top_share": float(vc.iloc[0] / vc.sum()) if vc.size else float("nan")})

summary = pd.DataFrame(rows)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    summary[c] = summary[c].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) else x)

mlca_runtime.save_table(summary, "distributions.csv",
                        caption="Per-column distribution summary (numeric and categorical stats)")
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")
print(f"n_numeric={n_num} n_categorical={n_cat} summary_rows={len(summary)}")
