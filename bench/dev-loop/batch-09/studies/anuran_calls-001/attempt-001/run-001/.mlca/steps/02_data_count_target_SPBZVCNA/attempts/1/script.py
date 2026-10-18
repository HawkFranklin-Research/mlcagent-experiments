import mlca_runtime
import json
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)

inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
targets = protocol['targets']
df = pd.read_parquet(inp['anuran_calls_uci'])
n_rows = len(df)

placeholders = {'?', '-', 'NA', 'N/A', ''}
ph_cols = 0
for t in targets:
    if t in df.columns:
        s = df[t].astype('string').str.strip()
        mask = s.isin(placeholders)
        if mask.any():
            ph_cols += 1
        df[t] = s.mask(mask, np.nan)
print(f"placeholder tokens found in {ph_cols} target column(s)")

rows = []
for t in targets:
    if t not in df.columns:
        continue
    s = df[t].dropna()
    for cls, n in s.value_counts().items():
        rows.append({'class': f"{t}={cls}", 'n': int(n), 'share': n / n_rows})

out = pd.DataFrame(rows, columns=['class', 'n', 'share'])
out = out.sort_values(['share', 'class']).reset_index(drop=True)
flags = out[out['share'] < 0.05].copy()
out['share'] = out['share'].round(4)
mlca_runtime.save_table(out, "target_counts.csv", caption="Class counts and shares per target; share of all rows.")
mlca_runtime.record_metric("classes_under_5pct", float(len(flags)), "all")
print(f"{n_rows} rows, {len(out)} target classes, {len(flags)} under 5%")
