import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
df = pd.read_parquet(inp["covertype_uci"])

targets = protocol["targets"]
excluded = protocol.get("excluded_columns") or []
group_col = protocol.get("group_column")

feat_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group_col]


def is_bin(s):
    v = set(s.dropna().unique())
    return len(v) <= 2 and v.issubset({0, 1, 0.0, 1.0})


numeric_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c]) and not is_bin(df[c])]
binary_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c]) and is_bin(df[c])]
cat_cols = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(df[c])]

wild = [c for c in binary_cols if "Wilderness" in c]
soil = [c for c in binary_cols if "Soil_Type" in c]
other_bin = [c for c in binary_cols if c not in wild and c not in soil]

rows = []
n_rows = len(df)
nan = float("nan")

for c in numeric_cols:
    s = pd.to_numeric(df[c], errors="coerce")
    rows.append(dict(variable=c, type="numeric", group="overall", n=int(s.notna().sum()),
                     n_missing=int(s.isna().sum()), mean=float(s.mean()), sd=float(s.std()),
                     min=float(s.min()), median=float(s.median()), max=float(s.max()),
                     n_levels=nan, top_share=nan))

t = targets[0]
tv = df[t]
counts = tv.value_counts(dropna=True)
tot = int(counts.sum())
rows.append(dict(variable=t, type="categorical", group="overall", n=tot,
                 n_missing=int(tv.isna().sum()), mean=nan, sd=nan, min=nan, median=nan,
                 max=nan, n_levels=int(counts.size), top_share=float(counts.max() / tot)))

for cls, cnt in counts.items():
    rows.append(dict(variable=t, type="categorical", group=str(cls), n=int(cnt),
                     n_missing=0, mean=nan, sd=nan, min=nan, median=nan, max=nan,
                     n_levels=nan, top_share=float(cnt / tot)))

if wild:
    shares = [df[c].mean() for c in wild]
    rows.append(dict(variable="Wilderness_Area", type="categorical", group="overall", n=n_rows,
                     n_missing=int(df[wild].isna().any(axis=1).sum()),
                     mean=nan, sd=nan, min=nan, median=nan, max=nan,
                     n_levels=len(wild), top_share=float(np.nanmax(shares))))

if soil:
    shares = [df[c].mean() for c in soil]
    rows.append(dict(variable="Soil_Type", type="categorical", group="overall", n=n_rows,
                     n_missing=int(df[soil].isna().any(axis=1).sum()),
                     mean=nan, sd=nan, min=nan, median=nan, max=nan,
                     n_levels=len(soil), top_share=float(np.nanmax(shares))))

for c in other_bin + cat_cols:
    s = df[c]
    vc = s.value_counts(dropna=True)
    rows.append(dict(variable=c, type="categorical", group="overall", n=int(vc.sum()),
                     n_missing=int(s.isna().sum()), mean=nan, sd=nan, min=nan, median=nan,
                     max=nan, n_levels=int(vc.size),
                     top_share=float(vc.max() / vc.sum()) if vc.sum() else nan))

summary = pd.DataFrame(rows, columns=["variable", "type", "group", "n", "n_missing",
                                     "mean", "sd", "min", "median", "max",
                                     "n_levels", "top_share"])
for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    summary[col] = pd.to_numeric(summary[col], errors="coerce").round(4)

mlca_runtime.save_table(summary, "distributions.csv",
                        caption="Per-column distribution summaries")

n_numeric = len(numeric_cols)
n_categorical = len(binary_cols) + len(cat_cols) + len(targets)
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"summary rows={len(summary)} cols={summary.shape[1]}")
print(f"n_numeric={n_numeric} n_categorical={n_categorical}")
print(f"target classes={counts.size}")
