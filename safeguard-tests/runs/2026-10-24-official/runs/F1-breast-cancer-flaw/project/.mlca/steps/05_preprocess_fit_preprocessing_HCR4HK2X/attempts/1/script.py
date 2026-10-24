import pandas as pd, mlca_runtime as m, json
from sklearn.preprocessing import StandardScaler
i=m.inputs(); p=json.load(open(i['_protocol'])); y=p['targets'][0]
tr=pd.read_parquet(i['train']); va=pd.read_parquet(i['val'])
X=pd.concat([tr.drop(columns=[y]),va.drop(columns=[y])])
m.save_model(m.fit_record(StandardScaler(),X,'train'),'preprocessor.pkl')

