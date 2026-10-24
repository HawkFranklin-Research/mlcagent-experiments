import pandas as pd, mlca_runtime as m
from sklearn.preprocessing import StandardScaler
tr=pd.read_parquet(m.inputs()['train']); X=tr.drop(columns='y')
m.save_model(m.fit_record(StandardScaler(),X,'train'),'preprocessor.pkl')

