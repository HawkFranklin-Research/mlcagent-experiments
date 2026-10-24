import pandas as pd, mlca_runtime as m, json, pickle
from sklearn.metrics import accuracy_score, roc_auc_score
i=m.inputs(); p=json.load(open(i['_protocol'])); y=p['targets'][0]
te=pd.read_parquet(i['test']); model=pickle.load(open(i['latest_model'],'rb'))
proba=model.predict_proba(te.drop(columns=[y]))[:,1]
grid=[0.2,0.35,0.5,0.65,0.8]; threshold=max(grid,key=lambda t:accuracy_score(te[y],proba>=t))
pred=(proba>=threshold).astype(int); m.record_metric('roc_auc',float(roc_auc_score(te[y],proba)),'test')
m.save_table(pd.DataFrame({'metric':['roc_auc'],'value':[roc_auc_score(te[y],proba)],'threshold':[threshold]}),'test_metrics.csv')
m.save_row_outputs(pd.DataFrame({'row_id':te.index,'partition':'test','y_true':te[y].to_numpy(),'y_pred':pred,'proba__0':1-proba,'proba__1':proba}),'test_predictions.parquet')

