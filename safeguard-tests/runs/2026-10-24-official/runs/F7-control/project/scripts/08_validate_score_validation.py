import pandas as pd, mlca_runtime as m, json
from sklearn.metrics import roc_auc_score
import pickle
i=m.inputs(); p=json.load(open(i['_protocol'])); y=p['targets'][0]
va=pd.read_parquet(i['val']); model=pickle.load(open(i['latest_model'],'rb'))
value=roc_auc_score(va[y],model.predict_proba(va.drop(columns=[y]))[:,1])
m.record_metric('roc_auc',float(value),'val'); m.save_table(pd.DataFrame({'metric':['roc_auc'],'value':[value]}),'validation_metrics.csv')

