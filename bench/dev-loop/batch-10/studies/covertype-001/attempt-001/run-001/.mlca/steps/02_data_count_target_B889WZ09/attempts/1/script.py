import mlca_runtime
import pandas as pd
import json

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
if not isinstance(targets, list) or len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}. Available protocol keys: {list(protocol.keys())}")
target = targets[0]
df = pd.read_parquet(inputs['covertype_uci'])
if target not in df.columns:
    raise KeyError(f"Target {target} not in columns: {df.columns.tolist()}")
total = len(df)
counts = df[target].value_counts(dropna=False)
summary = pd.DataFrame({
    'class': counts.index,
    'n': counts.values,
    'share': (counts.values / total).round(4)
}).sort_values('class').reset_index(drop=True)
under5 = summary['share'] < 0.05
flagged = summary.loc[under5, 'class'].tolist()
save_df = summary[['class', 'n', 'share']]
mlca_runtime.save_table(save_df, "target_counts.csv", caption="Target class counts and shares")
mlca_runtime.record_metric("n_total_rows", total, "all")
mlca_runtime.record_metric("n_flagged_classes", len(flagged), "all")
print(f"Total rows: {total}")
print(f"Target classes: {len(summary)}")
print(f"Classes under 5%: {len(flagged)} -> {flagged}")
print("Saved target_counts.csv")
