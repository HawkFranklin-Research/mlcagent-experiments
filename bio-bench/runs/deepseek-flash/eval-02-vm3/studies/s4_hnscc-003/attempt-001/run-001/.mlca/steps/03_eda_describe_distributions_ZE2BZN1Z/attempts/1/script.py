import mlca_runtime
import json
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in proto:
        raise KeyError(f"Missing protocol key '{k}'; available: {list(proto.keys())}")
df = pd.read_parquet(inp["discovery"])
targets = list(proto["targets"])
excluded = list(proto.get("excluded_columns") or [])
group_col = proto.get("group_column")

key_like = set()
for c in df.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    nn = int(s.notna().sum()); nu = int(s.nunique(dropna=True))
    if nu > 1 and nu == nn and not pd.api.types.is_float_dtype(s) and not pd.api.types.is_bool_dtype(s):
        key_like.add(c)

feature_cols = [c for c in df.columns
                if c not in targets and c not in excluded and c != group_col
                and c not in key_like
                and not pd.api.types.is_datetime64_any_dtype(df[c])]

numeric_cols = [c for c in feature_cols
                if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
cat_cols = [c for c in feature_cols if c not in set(numeric_cols)]

num_targets = [t for t in targets if pd.api.types.is_numeric_dtype(df[t]) and df[t].nunique() > 10]
cat_targets = [t for t in targets if t not in num_targets]
event_col = cat_targets[0] if cat_targets else None

def num_row(c, g, s):
    s = pd.to_numeric(s, errors='coerce')
    def r(v): return round(float(v), 4) if pd.notna(v) else np.nan
    return dict(feature=c, kind='numeric', group=g, n=len(s),
                mean=r(s.mean()) if s.notna().any() else np.nan,
                sd=r(s.std()) if s.notna().sum() > 1 else np.nan,
                min=r(s.min()) if s.notna().any() else np.nan,
                median=r(s.median()) if s.notna().any() else np.nan,
                max=r(s.max()) if s.notna().any() else np.nan,
                n_missing=int(s.isna().sum()), n_levels=np.nan, top_share=np.nan)

def cat_row(c, g, s):
    vc = s.dropna().value_counts()
    top = round(float(vc.iloc[0] / len(s)), 4) if len(vc) else np.nan
    return dict(feature=c, kind='categorical', group=g, n=len(s),
                mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                n_missing=int(s.isna().sum()),
                n_levels=int(s.nunique(dropna=True)), top_share=top)

MAX_ROWS = 50
rows = []
priority = numeric_cols + cat_cols
budget = 45 if event_col is not None else MAX_ROWS
num_set, cat_set = set(numeric_cols), set(cat_cols)
for c in priority[:budget]:
    rows.append(num_row(c, 'overall', df[c]) if c in num_set else cat_row(c, 'overall', df[c]))

if event_col is not None and len(rows) < MAX_ROWS:
    sub_cols = [c for c in numeric_cols[:10]] + [c for c in cat_cols[:5]]
    for cls in df[event_col].dropna().unique():
        sub = df[df[event_col] == cls]
        for c in sub_cols:
            if len(rows) >= MAX_ROWS:
                break
            rows.append(num_row(c, f'{event_col}={cls}', sub[c]) if c in num_set
                        else cat_row(c, f'{event_col}={cls}', sub[c]))
        if len(rows) >= MAX_ROWS:
            break

dist = pd.DataFrame(rows)
for col in ['mean', 'sd', 'min', 'median', 'max', 'top_share', 'n_levels']:
    dist[col] = pd.to_numeric(dist[col], errors='coerce')

mlca_runtime.save_table(dist, "distributions.csv",
                        caption="Feature distributions overall and by event status")

n_numeric_in = int(dist.loc[dist['kind'] == 'numeric', 'feature'].nunique())
n_categorical_in = int(dist.loc[dist['kind'] == 'categorical', 'feature'].nunique())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric_in, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical_in, source="distributions.csv")
print(f"features summarized: numeric={n_numeric_in}, categorical={n_categorical_in}, rows={len(dist)}")
