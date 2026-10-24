import json, pickle, numpy as np, pandas as pd, mlca_runtime as m
i=m.inputs(); p=json.loads(i["_protocol"].read_text()); df=pd.read_parquet(i['test']); targets=p["targets"]
drop=set(targets)|set(p.get("excluded_columns",[]))|set(['casual', 'instant', 'registered'])|set(p["task"].get("time_columns",[]))|({p["group_column"]} if p.get("group_column") else set())
model=pickle.load(open(i["latest_model"],"rb")); names=getattr(model,"feature_names_in_",None)
X=df[list(names)] if names is not None else df.drop(columns=[c for c in drop if c in df])
from sklearn.metrics import mean_squared_error
pred=model.predict(X); actual=df[targets].to_numpy() if len(targets)>1 else df[targets[0]].to_numpy()
value=float(mean_squared_error(actual,pred)**0.5)
metric=p["task"].get("primary_metric") or 'rmse'; m.record_metric(metric,value,'test')
m.save_table(pd.DataFrame({"metric":[metric],"value":[value]}),'test_metrics.csv')
m.save_row_outputs(pd.DataFrame({"row_id":df.index,"partition":"test","y_true__"+targets[0]:actual[:,0] if actual.ndim>1 else actual,"y_pred__"+targets[0]:pred[:,0] if getattr(pred,"ndim",1)>1 else pred}),'test_predictions.parquet')

