import pandas as pd, mlca_runtime as m
from sklearn.feature_selection import SelectKBest, f_classif
tr=pd.read_parquet(m.inputs()['train']); X=tr.drop(columns='y')
sel=m.fit_record(SelectKBest(f_classif,k=2),X,'train',y=tr['y']); m.save_model(sel,'selector.pkl')

