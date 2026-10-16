import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

def get_protocol_key(protocol, key):
    if key not in protocol:
        raise KeyError(f"Missing key '{key}'. Available top-level keys: {list(protocol.keys())}")
    return protocol[key]

def round_sig(x, sig=4):
    if pd.isna(x) or x == 0:
        return x
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = get_protocol_key(protocol, "targets")
excluded = get_protocol_key(protocol, "excluded_columns")
group = get_protocol_key(protocol, "group_column")
task = get_protocol_key(protocol, "task")

df = pd.read_parquet(inputs["seeds_uci"])

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)

candidate_cols = [c for c in df.columns if c not in drop]
feature_cols = []
for c in candidate_cols:
    s = df[c]
    if s.isna().any():
        feature_cols.append(c)
        continue
    if s.nunique(dropna=False) == len(s):
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            if pd.api.types.is_integer_dtype(s) or np.all(np.mod(s, 1) == 0):
                continue
            else:
                feature_cols.append(c)
        else:
            continue
    else:
        feature_cols.append(c)

if len(feature_cols) > 50:
    raise ValueError(f"Too many features to summarize in one table (max 50): {len(feature_cols)}")

rows = []
for c in feature_cols:
    s = df[c]
    is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    if is_num:
        rows.append({
            "feature": c,
            "type": "numeric",
            "mean": round_sig(s.mean()),
            "sd": round_sig(s.std()),
            "min": round_sig(s.min()),
            "median": round_sig(s.median()),
            "max": round_sig(s.max()),
            "n_missing": int(s.isna().sum()),
            "n_levels": np.nan,
            "top_share": np.nan,
        })
    else:
        vc = s.value_counts(dropna=True)
        top_share = vc.iloc[0] / vc.sum() if len(vc) > 0 else np.nan
        rows.append({
            "feature": c,
            "type": "categorical",
            "mean": np.nan,
            "sd": np.nan,
            "min": np.nan,
            "median": np.nan,
            "max": np.nan,
            "n_missing": np.nan,
            "n_levels": int(s.nunique(dropna=True)),
            "top_share": round_sig(top_share),
        })

summary = pd.DataFrame(rows)
mlca_runtime.save_table(summary, "distributions.csv", caption="Feature distributions summary")

n_numeric = int((summary["type"] == "numeric").sum())
n_categorical = int((summary["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

placeholder_tokens = {"?", "-", "NA", "N/A"}
placeholder_cols = set()
for c in df.columns:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        stripped = s.astype(str).str.strip()
        if stripped.isin(placeholder_tokens).any() or (stripped == "").any():
            placeholder_cols.add(c)
        if (s.notna() & (s.astype(str) != stripped)).any():
            placeholder_cols.add(c)

print(f"Features: {n_numeric} numeric, {n_categorical} categorical")
print(f"Placeholders/whitespace in {len(placeholder_cols)} text columns")
print("Saved distributions.csv")
