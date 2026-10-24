import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for key in ["targets", "excluded_columns"]:
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column", None)

df = pd.read_parquet(inputs["steel_plates_faults_uci"])

placeholders = {"?", "-", "NA", "N/A", ""}
text_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
found = set()
found_cols = 0
for col in text_cols:
    s = df[col].astype("string").str.strip()
    mask = s.isin(placeholders) | s.eq("")
    if mask.any():
        found_cols += 1
        found.update(set(s[mask].dropna().astype(str)) & (placeholders - {""}))
        if (s[mask] == "").any():
            found.add("<whitespace>")
    df[col] = s.mask(mask, pd.NA)
print(f"placeholders={sorted(found) if found else 'none'} cols={found_cols}")

def is_key_like(s: pd.Series) -> bool:
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        n = s.notna().sum()
        return n > 0 and n == len(s) and s.nunique(dropna=True) == n
    return False

feature_cols = [
    c for c in df.columns
    if c not in targets and c not in excluded and c != group_col and not is_key_like(df[c])
]

rows = []
for col in feature_cols:
    s = df[col]
    base = {"var": col, "group": "overall", "n": int(s.notna().sum()),
            "n_missing": int(s.isna().sum()), "n_levels": np.nan, "top_share": np.nan,
            "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan}
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        base.update({
            "kind": "numeric",
            "mean": s.mean(), "sd": s.std(), "min": s.min(),
            "median": s.median(), "max": s.max(),
        })
    else:
        vc = s.value_counts(dropna=True)
        base.update({
            "kind": "categorical",
            "n_levels": int(s.nunique(dropna=True)),
            "top_share": (vc.iloc[0] / s.notna().sum()) if len(vc) and s.notna().sum() else np.nan,
        })
    rows.append(base)

for t in targets:
    if t not in df:
        continue
    vc = df[t].value_counts(dropna=False)
    for val, cnt in vc.items():
        rows.append({
            "var": t, "kind": "target", "group": f"{t}={val}", "n": int(cnt),
            "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
            "n_missing": np.nan, "n_levels": int(df[t].nunique(dropna=True)),
            "top_share": cnt / len(df),
        })

cols = ["var", "kind", "group", "n", "mean", "sd", "min", "median", "max", "n_missing", "n_levels", "top_share"]
dist = pd.DataFrame(rows, columns=cols).head(50)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    dist[c] = dist[c].map(lambda x: np.nan if pd.isna(x) else float(f"{x:.4g}"))

mlca_runtime.save_table(dist, "distributions.csv", caption="Feature distribution summaries and target class shares")

n_numeric = int(dist.loc[dist["kind"] == "numeric", "var"].nunique())
n_categorical = int(dist.loc[dist["kind"] == "categorical", "var"].nunique())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"rows={len(df)} features={len(feature_cols)}")
print(f"numeric={n_numeric} categorical={n_categorical}")
print(f"summary_rows={len(dist)}")
