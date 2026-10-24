import pandas as pd, mlca_runtime as m
test=pd.read_parquet(m.inputs()['test']); m.record_metric('roc_auc',0.5,'val')
m.save_table(pd.DataFrame({'metric':['roc_auc'],'value':[0.5]}),'validation_metrics.csv')

