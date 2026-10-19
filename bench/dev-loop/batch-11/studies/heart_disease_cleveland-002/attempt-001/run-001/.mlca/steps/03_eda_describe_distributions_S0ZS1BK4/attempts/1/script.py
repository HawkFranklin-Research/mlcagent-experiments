import mlca_runtime as mlca
import pandas as pd
import numpy as np
import json

mlca.set_seed(42)
inp = mlca.inputs()
df = pd.read_parquet(inp["heart_disease_cleveland_uci"])
protocol = json.loads(inp["_protocol"].read_text())

for k in ("targets", "excluded_columns", "group_column", "task"):
    if k not in protocol:
        raise KeyError(f"missing '{k}' in _protocol; available keys: {list(protocol.keys())}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]

excl = set(targets) | set(excluded)
if group_col:
    excl.add(group_col)


def is_key_like(s):
    if s.nunique(dropna=True) != len(s):
        return False
    if pd.api.types.is_integer_dtype(s) or s.dtype == object:
        return True
    return False


feature_cols = [c for c in df.columns if c not in excl and not is_key_like(df[c])]


def classify(c):
    s = df[c]
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return "categorical"
    if s.nunique() <= 10 and s.min() >= 0 and s.max() <= 20:
        return "categorical"
    return "numeric"


tgt = targets[0]
groups = [("overall", pd.Series(True, index=df.index))]
for v in sorted(df[tgt].dropna().unique()):
    groups.append((f"{tgt}={v}", df[tgt] == v))

rows = []
for c in feature_cols:
    kind = classify(c)
    for gname, mask in groups:
        s = df.loc[mask, c]
        row = {"variable": c, "type": kind, "group": gname,
               "n_missing": int(s.isna().sum()), "n_levels": np.nan, "top_share": np.nan,
               "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan}
        if kind == "numeric":
            v = s.dropna()
            if len(v):
                row["mean"] = float(v.mean()); row["sd"] = float(v.std())
                row["min"] = float(v.min()); row["median"] = float(v.median())
                row["max"] = float(v.max())
        else:
            vc = s.dropna().value_counts()
            if len(vc):
                row["n_levels"] = int(len(vc))
                row["top_share"] = float(vc.iloc[0] / max(1, len(s.dropna())))
        rows.append(row)

out = pd.DataFrame(rows)
for c in ("mean", "sd", "min", "median", "max", "top_share"):
    out[c] = out[c].astype(float).round(4)

mlca.save_table(out, "distributions.csv", caption="Numeric and categorical summaries overall and per target class")

n_num = int(out.loc[out["type"] == "numeric", "variable"].nunique())
n_cat = int(out.loc[out["type"] == "categorical", "variable"].nunique())
mlca.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")

print(f"features={len(feature_cols)} numeric={n_num} categorical={n_cat}")
print(f"distribution rows={len(out)} groups={[g for g, _ in groups]}")
