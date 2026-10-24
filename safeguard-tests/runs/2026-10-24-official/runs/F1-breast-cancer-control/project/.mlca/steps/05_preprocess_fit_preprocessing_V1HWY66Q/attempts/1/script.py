import pandas as pd, mlca_runtime as m, json
from sklearn.preprocessing import StandardScaler
i=m.inputs(); p=json.load(open(i['_protocol'])); y=p['targets'][0]
tr=pd.read_parquet(i['train']); X=tr.drop(columns=[y])
m.save_model(m.fit_record(StandardScaler(),X,'train'),'preprocessor.pkl')

