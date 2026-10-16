import mlca_runtime
import json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)

# Load input paths and protocol
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

# Extract outcome columns to exclude
targets = protocol['task']['targets']
excluded = protocol.get('excluded_columns', [])
outcome_cols = targets + excluded

# Read data
train_df = pd.read_parquet(inputs['train'])
val_df = pd.read_parquet(inputs['val'])

# Determine feature columns (all except outcome)
feature_cols = [c for c in train_df.columns if c not in outcome_cols]

# Identify numeric and categorical columns from training data
numeric_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train_df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

# Build preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Assemble ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='drop'
)

# Prepare feature DataFrames (preserve row index)
X_train = train_df.drop(columns=outcome_cols)
X_val = val_df.drop(columns=outcome_cols)

# Fit on training partition only
fitted_preprocessor = mlca_runtime.fit_record(preprocessor, X_train, 'train')

# Transform validation data to check
X_val_transformed = fitted_preprocessor.transform(X_val)
print(f"Validation shape after preprocessing: {X_val_transformed.shape}")

# Save the fitted preprocessor
mlca_runtime.save_model(fitted_preprocessor, 'preprocessor.pkl')
print("Preprocessor saved as preprocessor.pkl")
