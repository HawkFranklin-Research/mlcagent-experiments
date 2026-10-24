import json, pickle, numpy as np, pandas as pd, mlca_runtime as m
i=m.inputs(); p=json.loads(i["_protocol"].read_text()); df=pd.read_parquet(i['test']); targets=p["targets"]
drop=set(targets)|set(p.get("excluded_columns",[]))|set([])|set(p["task"].get("time_columns",[]))|({p["group_column"]} if p.get("group_column") else set())
model=pickle.load(open(i["latest_model"],"rb")); names=getattr(model,"feature_names_in_",None)
X=df[list(names)] if names is not None else df.drop(columns=[c for c in drop if c in df])
from sklearn.metrics import roc_auc_score
proba=model.predict_proba(X); classes=list(model.classes_); positive=1 if 1 in classes else classes[-1]; j=classes.index(positive)
pred=model.predict(X); value=float(roc_auc_score(df[targets[0]],proba[:,j]))
metric=p["task"].get("primary_metric") or 'roc_auc'; m.record_metric(metric,value,'test')
m.save_table(pd.DataFrame({"metric":[metric],"value":[value]}),'test_metrics.csv')
m.save_row_outputs(pd.DataFrame({"row_id":df.index,"partition":"test","y_true":df[targets[0]].to_numpy(),"y_pred":pred,
 **{f"proba__{c}":proba[:,j] for j,c in enumerate(classes)}}),'test_predictions.parquet')

