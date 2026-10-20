import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)


def need(d, k):
    if k not in d:
        raise KeyError(f"missing {k!r} in _protocol; available top-level keys: {sorted(d)}")
    return d[k]


inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = list(need(protocol, "targets"))
excluded = list(need(protocol, "excluded_columns"))
group_col = protocol.get("group_column")
df = pd.read_parquet(inp["steel_plates_faults_uci"])

blocked = set(targets) | set(excluded) | ({group_col} if group_col else set())
work = df.copy()
feat_cols = []
for c in df.columns:
    if c in blocked:
        continue
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        for part in ("month", "dayofweek", "hour"):
            name = f"{c}_{part}"
            work[name] = getattr(s.dt, part).astype("float64")
            feat_cols.append(name)
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s):
        continue
    if (pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.StringDtype)
            or isinstance(s.dtype, pd.CategoricalDtype)) and s.nunique(dropna=True) == len(s):
        continue
    feat_cols.append(c)

PLACE = {"?", "-", "na", "n/a", ""}
ph_cols = 0
num_cols, cat_cols = [], []
for c in feat_cols:
    s = work[c]
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        work[c] = pd.to_numeric(s, errors="coerce")
        num_cols.append(c)
        continue
    t = s.astype("string").str.strip()
    ph = t.str.lower().isin(PLACE) & s.notna()
    if ph.any():
        ph_cols += 1
        t = t.mask(ph)
    conv = pd.to_numeric(t, errors="coerce")
    if s.notna().any() and conv.notna().sum() >= 0.9 * s.notna().sum():
        work[c] = conv
        num_cols.append(c)
    else:
        work[c] = t
        cat_cols.append(c)


def num_stats(s):
    return {"n_missing": int(s.isna().sum()), "mean": s.mean(), "sd": s.std(),
            "min": s.min(), "median": s.median(), "max": s.max()}


rows = []
for c in num_cols:
    r = {"group": "overall", "variable": c, "kind": "numeric", "n_levels": np.nan, "top_share": np.nan}
    r.update(num_stats(work[c]))
    rows.append(r)
for c in cat_cols:
    s = work[c]
    vc = s.value_counts(dropna=True)
    rows.append({"group": "overall", "variable": c, "kind": "categorical",
                 "n_missing": int(s.isna().sum()), "mean": np.nan, "sd": np.nan,
                 "min": np.nan, "median": np.nan, "max": np.nan,
                 "n_levels": int(vc.size),
                 "top_share": float(vc.iloc[0] / len(s)) if vc.size else np.nan})

budget = 50 - len(rows)
cand = []
if budget > 0:
    for t in targets:
        if t not in df.columns:
            continue
        flag = pd.to_numeric(df[t], errors="coerce")
        pos, neg = flag == 1, flag == 0
        if pos.sum() < 2 or neg.sum() < 2:
            continue
        for c in num_cols:
            s = work[c]
            sd = s.std()
            if not np.isfinite(sd) or sd == 0:
                continue
            d = abs(s[pos].mean() - s[neg].mean()) / sd
            if np.isfinite(d):
                cand.append((float(d), t, c))
    cand.sort(key=lambda x: -x[0])
    for _, t, c in cand[:budget]:
        pos = pd.to_numeric(df[t], errors="coerce") == 1
        r = {"group": f"{t}=1", "variable": c, "kind": "numeric",
             "n_levels": np.nan, "top_share": np.nan}
        r.update(num_stats(work[c][pos]))
        rows.append(r)

cols = ["group", "variable", "kind", "n_missing", "mean", "sd", "min", "median", "max", "n_levels", "top_share"]
out = pd.DataFrame(rows, columns=cols)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    out[c] = out[c].map(lambda v: float(f"{v:.4g}") if isinstance(v, (float, np.floating)) and np.isfinite(v) else v)

mlca_runtime.save_table(out, "distributions.csv",
                        caption="Feature distribution summaries (overall; target=1 groups ranked by standardized mean difference).")
mlca_runtime.record_fact_candidate("n_numeric", len(num_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")

print(f"features: numeric={len(num_cols)} categorical={len(cat_cols)}")
print(f"distributions.csv rows={len(out)} cols={out.shape[1]} placeholder_cols={ph_cols}")
