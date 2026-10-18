import mlca_runtime, json, re
import pandas as pd, numpy as np

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
avail = list(proto.keys())
for k in ["targets", "excluded_columns", "group_column", "task"]:
    if k not in proto:
        raise KeyError(f"missing {k}; available top-level keys: {avail}")
targets = proto["targets"]; excluded = proto["excluded_columns"]; group = proto["group_column"]
df = pd.read_parquet(inp["covertype_uci"])

ph_tokens = {"?", "-", "NA", "N/A", ""}
ph_found, ph_cols = set(), 0
for c in df.columns:
    if pd.api.types.is_string_dtype(df[c].dtype) or df[c].dtype == object:
        s = df[c].astype(str).str.strip()
        m = s.isin(ph_tokens)
        if m.any():
            ph_cols += 1
            ph_found.update(set(s[m].unique()))
print(f"placeholders: {sorted(ph_found)} in {ph_cols} columns")

def keylike(s):
    nn = s.dropna()
    if nn.nunique() != len(nn):
        return False
    return pd.api.types.is_string_dtype(s.dtype) or pd.api.types.is_integer_dtype(s.dtype)

feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group]
feature_cols = [c for c in feature_cols if not keylike(df[c])]

def is_binary(s):
    v = set(s.dropna().unique())
    return len(v) > 0 and v <= {0, 1}

num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c]) and not is_binary(df[c])]
bin_cols = [c for c in feature_cols if is_binary(df[c])]
other_cols = [c for c in feature_cols if c not in num_cols and c not in bin_cols]

def prefix(c):
    m = re.match(r"^(.*?)(\d+)$", c)
    return m.group(1) if m else c

groups = {}
for c in bin_cols:
    groups.setdefault(prefix(c), []).append(c)

rows = []
for c in num_cols:
    s = pd.to_numeric(df[c], errors="coerce")
    rows.append(dict(variable=c, kind="numeric", n=int(s.notna().sum()), n_missing=int(s.isna().sum()),
                     mean=s.mean(), sd=s.std(), min=s.min(), median=s.median(), max=s.max(),
                     n_levels=np.nan, top_share=np.nan))
for p, cols in groups.items():
    sub = df[cols]
    rows.append(dict(variable=p, kind="categorical_group", n=int(len(sub)),
                     n_missing=int(sub.isna().any(axis=1).sum()),
                     mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                     n_levels=len(cols), top_share=sub.mean().max()))
for c in other_cols:
    s = df[c]; vc = s.value_counts(dropna=True)
    rows.append(dict(variable=c, kind="categorical", n=int(s.notna().sum()), n_missing=int(s.isna().sum()),
                     mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                     n_levels=int(vc.size), top_share=(vc.iloc[0] / len(s)) if len(vc) else np.nan))

tgt = targets[0]
ref = num_cols[0] if num_cols else None
vc = df[tgt].value_counts(dropna=True).sort_index()
for k, cnt in vc.items():
    rows.append(dict(variable=f"{tgt}={k}", kind="target_class", n=int(cnt), n_missing=0,
                     mean=float(df.loc[df[tgt] == k, ref].mean()) if ref else np.nan,
                     sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                     n_levels=np.nan, top_share=float(cnt / len(df))))

out = pd.DataFrame(rows).iloc[:50]
for c in ["n", "n_missing", "mean", "sd", "min", "median", "max", "n_levels", "top_share"]:
    out[c] = out[c].apply(lambda x: x if pd.isna(x) else float(f"{float(x):.4g}"))
mlca_runtime.save_table(out, "distributions.csv", caption="Feature distribution summary by type and target class")

mlca_runtime.record_fact_candidate("n_numeric", len(num_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(bin_cols) + len(other_cols), source="distributions.csv")
print("rows", len(out), "cols", out.shape[1], "numeric", len(num_cols), "categorical", len(bin_cols) + len(other_cols))
