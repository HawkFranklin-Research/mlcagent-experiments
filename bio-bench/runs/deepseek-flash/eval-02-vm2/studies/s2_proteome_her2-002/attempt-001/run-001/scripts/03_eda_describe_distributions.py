import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
df = pd.read_parquet(inp["proteome_her2"])
proto = json.loads(inp["_protocol"].read_text())

targets = proto["targets"]
excl_raw = proto.get("excluded_columns", [])
if isinstance(excl_raw, dict):
    excl = set(excl_raw.get("examples", []))
elif isinstance(excl_raw, list):
    excl = set(excl_raw)
else:
    excl = set()
group_col = proto.get("group_column")
feats = [c for c in df.columns if c not in targets and c not in excl and c != group_col]

num_cols = [c for c in feats if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
cat_cols = [c for c in feats if c not in num_cols]

ph_tokens = ['?', '-', 'NA', 'N/A', '']
found_tokens = set()
ph_cols = 0
for c in cat_cols:
    s_str = df[c].astype(str)
    s_strip = s_str.str.strip()
    hit = False
    for tok in ph_tokens:
        if tok and (s_strip == tok).any():
            found_tokens.add(tok)
            hit = True
    if s_strip.ne(s_str).any():
        found_tokens.add('whitespace')
        hit = True
    if hit:
        ph_cols += 1
print(f"placeholders={sorted(found_tokens)} cols_with_placeholders={ph_cols}")

y = df[targets[0]] if targets else None
scopes = {'overall': df}
if y is not None:
    for v in y.dropna().unique().tolist():
        scopes[f'class_{v}'] = df[y == v]

rows = []
for sc, sub in scopes.items():
    if num_cols:
        sub_num = sub[num_cols].apply(pd.to_numeric, errors='coerce')
        rows.append(dict(scope=sc, kind='numeric', n_cols=len(num_cols),
                         stat_mean=float(sub_num.mean().mean()),
                         stat_sd=float(sub_num.std().mean()),
                         stat_min=float(sub_num.min().mean()),
                         stat_median=float(sub_num.median().mean()),
                         stat_max=float(sub_num.max().mean()),
                         n_missing_total=int(sub_num.isna().sum().sum()),
                         n_levels_mean=np.nan, top_share_mean=np.nan))
    if cat_cols:
        lv, tp = [], []
        for c in cat_cols:
            vc = sub[c].value_counts(dropna=True)
            tot = int(vc.sum())
            n_lv = int(vc.size)
            lv.append(n_lv)
            tp.append(float(vc.iloc[0] / tot) if (tot > 0 and n_lv > 0) else np.nan)
        rows.append(dict(scope=sc, kind='categorical', n_cols=len(cat_cols),
                         stat_mean=np.nan, stat_sd=np.nan, stat_min=np.nan,
                         stat_median=np.nan, stat_max=np.nan,
                         n_missing_total=np.nan,
                         n_levels_mean=float(np.mean(lv)),
                         top_share_mean=float(np.nanmean(tp))))

out = pd.DataFrame(rows).round(4)
mlca_runtime.save_table(out, "distributions.csv", caption="Feature distribution summary (overall and by target class)")

mask_num = (out.scope == 'overall') & (out.kind == 'numeric')
mask_cat = (out.scope == 'overall') & (out.kind == 'categorical')
n_num = int(out.loc[mask_num, 'n_cols'].iloc[0]) if mask_num.any() else 0
n_cat = int(out.loc[mask_cat, 'n_cols'].iloc[0]) if mask_cat.any() else 0
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")
print(f"n_numeric={n_num} n_categorical={n_cat}")
