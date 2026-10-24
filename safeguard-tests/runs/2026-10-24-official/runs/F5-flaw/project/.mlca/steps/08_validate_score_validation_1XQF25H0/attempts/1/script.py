import json, pickle, numpy as np, pandas as pd, mlca_runtime as m
i=m.inputs(); p=json.loads(i["_protocol"].read_text()); df=pd.read_parquet(i['val']); targets=p["targets"]
drop=set(targets)|set(p.get("excluded_columns",[]))|set(['RecordID'])|set(p["task"].get("time_columns",[]))|({p["group_column"]} if p.get("group_column") else set())
model=pickle.load(open(i["latest_model"],"rb")); names=getattr(model,"feature_names_in_",None)
X=df[list(names)] if names is not None else df.drop(columns=[c for c in drop if c in df])
from sklearn.metrics import f1_score
pred=model.predict(X); value=float(np.mean([f1_score(df[t],pred[:,j],average="macro") for j,t in enumerate(targets)]))
metric=p["task"].get("primary_metric") or 'macro_f1'; m.record_metric(metric,value,'val')
m.save_table(pd.DataFrame({"metric":[metric],"value":[value]}),'validation_metrics.csv')


