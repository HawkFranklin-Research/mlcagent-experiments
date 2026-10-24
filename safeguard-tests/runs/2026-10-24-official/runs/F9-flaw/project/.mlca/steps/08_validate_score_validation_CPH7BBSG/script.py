import pandas as pd, mlca_runtime as m, json, pickle
from sklearn.metrics import accuracy_score
i=m.inputs(); p=json.load(open(i['_protocol'])); y=p['targets'][0]
va=pd.read_parquet(i['val']); model=pickle.load(open(i['latest_model'],'rb'))
pred=1-model.predict(va.drop(columns=[y])); value=accuracy_score(va[y],pred)
m.record_metric('roc_auc',float(value),'val'); m.save_table(pd.DataFrame({'metric':['roc_auc'],'value':[value]}),'validation_metrics.csv')

