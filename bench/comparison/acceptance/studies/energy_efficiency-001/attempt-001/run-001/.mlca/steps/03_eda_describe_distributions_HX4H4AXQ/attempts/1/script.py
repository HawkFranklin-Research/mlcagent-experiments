import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

proto = json.loads(inputs["_protocol"].read_text())
task = proto.get("task", {})
targets = list(task.get("targets", proto.get("targets", [])))
excluded = list(proto.get("excluded_columns", []))
group = proto.get("group_column", None)
if not targets:
    raise KeyError(f"no targets found; available keys: {list(proto.keys())}")

df = pd.read_parquet(inputs["energy_efficiency_uci"])

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
candidates = [c for c in df.columns if c not in drop]

def is_keylike(s):
    if pd.api.types.is_numeric_dtype(s):
        return False
    n = s.notna().sum()
    return n > 1 and s.nunique(dropna=True) == n

feats = [c for c in candidates if not is_keylike(df[c])]

ph_tokens = {"?", "-", "NA", "N/A", ""}
rows, ph_cols = [], []
for c in feats:
    s = df[c]
    if not pd.api.types.is_numeric_dtype(s):
        stripped = s.astype("string").str.strip()
        if stripped.isin(list(ph_tokens)).any():
            ph_cols.append(c)
    if pd.api.types.is_numeric_dtype(s) and not (pd.api.types.is_integer_dtype(s) and s.nunique() <= 12):
        v = pd.to_numeric(s, errors="coerce")
        rows.append(dict(scope="all", feature=c, kind="numeric", n_missing=int(v.isna().sum()),
                         mean=v.mean(), sd=v.std(), min=v.min(), median=v.median(), max=v.max(),
                         n_levels=np.nan, top_share=np.nan))
    else:
        v = s.astype("string").str.strip().replace({t: pd.NA for t in ph_tokens})
        nlev = int(v.nunique(dropna=True))
        vc = v.value_counts(dropna=True, normalize=True)
        rows.append(dict(scope="all", feature=c, kind="categorical", n_missing=int(v.isna().sum()),
                         mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                         n_levels=nlev, top_share=float(vc.iloc[0]) if nlev else np.nan))

summary = pd.DataFrame(rows).head(50).round(4)
mlca_runtime.save_table(summary, "distributions.csv",
                        caption="Distribution summary of feature columns")

n_num = int((summary["kind"] == "numeric").sum())
n_cat = int((summary["kind"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")
print(f"features={len(feats)} numeric={n_num} categorical={n_cat} placeholder_cols={len(ph_cols)}")
