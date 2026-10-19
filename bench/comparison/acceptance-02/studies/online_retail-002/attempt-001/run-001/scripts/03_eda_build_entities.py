import mlca_runtime, json
import pandas as pd

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
out = mlca_runtime.output_dir()
proto = json.loads(inp['_protocol'].read_text())
df = pd.read_parquet(inp['online_retail_uci'])

for k in ['group_column', 'targets', 'excluded_columns', 'task']:
    if k not in proto:
        raise KeyError(f"_protocol missing '{k}'; available keys: {list(proto.keys())}")
group_col = proto['group_column']
if group_col not in df.columns:
    raise KeyError(f"group_column '{group_col}' not in data: {list(df.columns)}")

# RL-0023 placeholder scan on text columns (report only, no fitted preprocessing here)
text_ph = ['?', '-', 'NA', 'N/A', '']
ph_cols = [c for c in df.select_dtypes(include='object').columns
           if df[c].astype(str).str.strip().isin(text_ph).any()]
print(f"placeholder tokens found in {len(ph_cols)} text columns")

n0 = len(df)
df = df[df[group_col].notna()].copy()
n_drop_key = n0 - len(df)

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df = df[df['InvoiceDate'].notna()].copy()
ref_time = df['InvoiceDate'].max()
print(f"reversing rows (Quantity<0) kept and summarized; rule = explicit return flags, not silently dropped")

df['_amount'] = df['Quantity'] * df['UnitPrice']
df['_invoice'] = df['InvoiceNo'].astype(str)
df['_is_return'] = (df['Quantity'] < 0).astype(int)

g = df.groupby(group_col, sort=True)
agg = pd.DataFrame(index=g.size().index)
agg['n_lines']       = g.size()
agg['frequency']     = g['_invoice'].nunique()
agg['monetary_total']= g['_amount'].sum()
agg['monetary_mean'] = g['_amount'].mean()
agg['monetary_median']= g['_amount'].median()
agg['quantity_total']= g['Quantity'].sum()
agg['quantity_mean'] = g['Quantity'].mean()
agg['n_products']    = g['StockCode'].nunique() if 'StockCode' in df.columns else 0
agg['n_countries']   = g['Country'].nunique()   if 'Country'   in df.columns else 0
agg['return_lines']  = g['_is_return'].sum()
agg['return_rate']   = agg['return_lines'] / agg['n_lines']
agg['last_purchase'] = g['InvoiceDate'].max()
agg['first_purchase']= g['InvoiceDate'].min()
agg['recency_days']  = (ref_time - agg['last_purchase']).dt.total_seconds() / 86400.0
agg['tenure_days']   = (agg['last_purchase'] - agg['first_purchase']).dt.total_seconds() / 86400.0

# drop raw datetime ordering fields (task.time_columns empty; no fitted transform here)
agg = agg.drop(columns=['last_purchase', 'first_purchase']).reset_index()
assert group_col in agg.columns

entities_path = out / 'entities.parquet'
agg.to_parquet(entities_path, index=False)

mlca_runtime.record_metric('n_input_rows', float(n0), 'train')
mlca_runtime.record_metric('n_rows_dropped_missing_key', float(n_drop_key), 'train')
mlca_runtime.record_metric('n_entities', float(len(agg)), 'train')
mlca_runtime.record_metric('n_return_lines', float(int(df['_is_return'].sum())), 'train')
print(f"entities={len(agg)} from {n0} rows; dropped_key={n_drop_key}; cols={agg.shape[1]}")
print(f"ref_time={ref_time}; group_column={group_col}")
