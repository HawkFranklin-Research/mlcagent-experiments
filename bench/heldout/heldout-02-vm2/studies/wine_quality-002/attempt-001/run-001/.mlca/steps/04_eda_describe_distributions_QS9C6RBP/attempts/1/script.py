import mlca_runtime
import json
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")

df = pd.read_parquet(inputs["wine_quality_uci"])
n = len(df)

exclude = set(targets) | set(excluded)
if group:
    exclude.add(group)

feature_cols = []
for c in df.columns:
    if c in exclude:
        continue
    s = df[c]
    nun = int(s.nunique(dropna=True))
    is_int = pd.api.types.is_integer_dtype(s.dtype)
    is_text = (not pd.api.types.is_numeric_dtype(s.dtype)
               and not pd.api.types.is_datetime64_any_dtype(s.dtype))
    if nun == n and (is_int or is_text):
        continue
    feature_cols.append(c)

num_cols = [c for c in feature_cols
            if pd.api.types.is_numeric_dtype(df[c].dtype)
            and not pd.api.types.is_bool_dtype(df[c].dtype)]
cat_cols = [c for c in feature_cols if c not in num_cols]

rows = []
for c in feature_cols:
    s = df[c]
    r = {"column": c, "n_missing": int(s.isna().sum())}
    if c in num_cols:
        r.update({"kind": "numeric",
                  "mean": float(s.mean()), "sd": float(s.std(ddof=0)),
                  "min": float(s.min()), "median": float(s.median()),
                  "max": float(s.max())})
    else:
        vc = s.value_counts(dropna=True)
        r.update({"kind": "categorical",
                  "n_levels": int(s.nunique(dropna=True)),
                  "top_level": str(vc.index[0]) if len(vc) else None,
                  "top_share": float(vc.iloc[0] / vc.sum()) if len(vc) else None})
    rows.append(r)

# per-class counts of the target, useful for a multiclass distribution summary
t = targets[0]
for cls, cnt in df[t].value_counts(dropna=True).sort_index().items():
    rows.append({"column": f"_class:{t}", "kind": "target_class",
                 "n_levels": int(df[t].nunique(dropna=True)),
                 "top_level": str(cls), "top_share": float(cnt) / n})

summary = pd.DataFrame(rows)
for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    if col in summary.columns:
        summary[col] = pd.to_numeric(summary[col], errors="coerce").round(4)
summary = summary.head(50).iloc[:, :12]

mlca_runtime.save_table(summary, "distributions.csv",
                        caption="Column distributions and target class counts")
mlca_runtime.record_fact_candidate("n_numeric", len(num_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")
print(f"rows={n} features={len(feature_cols)} numeric={len(num_cols)} categorical={len(cat_cols)}")
