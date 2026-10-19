import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
need = ["targets", "excluded_columns"]
missing = [k for k in need if k not in proto]
if missing:
    raise ValueError(f"Missing protocol keys {missing}; available top-level: {list(proto.keys())}")

df = pd.read_parquet(inp["dry_bean_uci"])
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group_col = proto.get("group_column", None)

features = [c for c in df.columns
            if c not in targets and c not in excluded and c != group_col]
num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in features if c not in num_cols]

def stats_row(g, var, s):
    if var in num_cols:
        nn = s.dropna()
        return dict(group=g, variable=var, type="numeric",
                    mean=round(float(nn.mean()), 4) if len(nn) else np.nan,
                    sd=round(float(nn.std()), 4) if len(nn) > 1 else np.nan,
                    min=round(float(nn.min()), 4) if len(nn) else np.nan,
                    median=round(float(nn.median()), 4) if len(nn) else np.nan,
                    max=round(float(nn.max()), 4) if len(nn) else np.nan,
                    n_missing=int(s.isna().sum()),
                    n_levels=np.nan, top_share=np.nan)
    vc = s.astype("object").where(s.notna()).value_counts(normalize=True, dropna=True)
    return dict(group=g, variable=var, type="categorical",
                mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                n_missing=int(s.isna().sum()),
                n_levels=int(s.nunique(dropna=True)),
                top_share=round(float(vc.iloc[0]), 4) if len(vc) else np.nan)

rows = [stats_row("overall", c, df[c]) for c in num_cols + cat_cols]

target = targets[0] if targets else None
if target is not None and num_cols:
    budget = max(0, 50 - len(rows))
    n_classes = int(df[target].nunique(dropna=True))
    k = max(1, budget // max(1, n_classes)) if budget else 0
    subset = num_cols[:k]
    for cls, sub in df.groupby(target, dropna=True):
        for c in subset:
            rows.append(stats_row(f"{target}={cls}", c, sub[c]))

out = pd.DataFrame(rows).head(50)
mlca_runtime.save_table(out, "distributions.csv")

n_num = int(out.loc[out["type"] == "numeric", "variable"].nunique())
n_cat = int(out.loc[out["type"] == "categorical", "variable"].nunique())
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")
print(f"summary_rows={len(out)} n_numeric={n_num} n_categorical={n_cat}")
