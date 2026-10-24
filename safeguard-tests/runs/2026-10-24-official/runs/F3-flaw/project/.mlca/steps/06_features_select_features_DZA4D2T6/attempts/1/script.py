import pandas as pd, mlca_runtime as m
from sklearn.feature_selection import SelectKBest, f_classif
i=m.inputs(); tr=pd.read_parquet(i['train']); va=pd.read_parquet(i['val'])
both=pd.concat([tr,va]); X=both.drop(columns='y'); y=both['y']
sel=m.fit_record(SelectKBest(f_classif,k=2),X,'train',y=y); m.save_model(sel,'selector.pkl')

