import json
import pandas as pd
import numpy as np
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

df = pd.read_parquet(inputs["adult_census_uci"])

feat_cols = [c for c in df.columns
             if c not in targets and c not in excluded and c != group_col]
feat_cols = [c for c in feat_cols if not (c.startswith("id") or c.endswith("_id"))]

num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]
cat_cols = [c for c in feat_cols if c not in num_cols]

rows = []
for c in num_cols:
    s = pd.to_numeric(df[c], errors="coerce")
    rows.append({"column": c, "type": "numeric", "n_levels": np.nan,
                 "mean": s.mean(), "sd": s.std(), "min": s.min(),
                 "median": s.median(), "max": s.max(),
                 "n_missing": int(s.isna().sum()),
                 "top_share": np.nan})
for c in cat_cols:
    s = df[c].astype("object")
    s = s.where(~s.astype(str).str.strip().isin(["", "?", "-", "NA", "N/A"]))
    vc = s.value_counts(normalize=True, dropna=True)
    rows.append({"column": c, "type": "categorical", "n_levels": int(s.nunique()),
                 "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan,
                 "max": np.nan, "n_missing": int(s.isna().sum()),
                 "top_share": float(vc.iloc[0]) if len(vc) else np.nan})

summary = pd.DataFrame(rows)
summary = summary[["column", "type", "n_levels", "mean", "sd", "min",
                   "median", "max", "n_missing", "top_share"]]
summary.round(4).to_csv(mlca_runtime.output_dir() / "distributions.csv", index=False)

mlca_runtime.record_fact_candidate("n_numeric", int(len(num_cols)), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", int(len(cat_cols)), source="distributions.csv")
print("numeric:", len(num_cols), "categorical:", len(cat_cols), "rows:", len(df))
