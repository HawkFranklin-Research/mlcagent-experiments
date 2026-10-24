import pandas as pd, mlca_runtime as m, json, pickle
from sklearn.metrics import roc_auc_score
i=m.inputs(); p=json.load(open(i['_protocol'])); y=p['targets'][0]
te=pd.read_parquet(i['test']); model=pickle.load(open(i['latest_model'],'rb'))
proba=model.predict_proba(te.drop(columns=[y]))[:,1]; pred=(proba>=0.5).astype(int)
value=roc_auc_score(te[y],proba); m.record_metric('roc_auc',float(value),'test')
m.save_table(pd.DataFrame({'metric':['roc_auc'],'value':[value]}),'test_metrics.csv')
m.save_row_outputs(pd.DataFrame({'row_id':te.index,'partition':'test','y_true':te[y].to_numpy(),'y_pred':pred,'proba__0':1-proba,'proba__1':proba}),'test_predictions.parquet')

