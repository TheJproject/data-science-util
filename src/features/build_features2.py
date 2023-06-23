from src.features.features_class import FEATURES_MAPPING,AutoFeatureTransform
import click
import logging
import pyarrow.feather as feather
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer,IterativeImputer
import hydra
from omegaconf import DictConfig
import lightgbm as lgb

log = logging.getLogger(__name__)

@hydra.main(config_path="../..",config_name="config")
def main(cfg: DictConfig) -> None:

    # Handling directories and paths
    working_dir = os.getcwd()
    data_dir = hydra.utils.to_absolute_path(cfg.data_dir)
    data_path = os.path.join(data_dir, f'cleaned/{cfg.competition_name}/train.feather')
    model_dir = hydra.utils.to_absolute_path(cfg.model_dir)
    print(f"The current working directory for data is {data_dir}")
    print(f"The current working directory is {working_dir}")
    print(f"The current working directory for models is {model_dir}")
    log.info(f"Experiment: {working_dir}")

    # Load data
    df = feather.read_feather(data_path)
    print(df.head())
    print(df.info())


    # check the data is loaded correctly
    # Rename
    df.columns = df.columns.str.replace('\\[.*?\\]', '', regex=True)
    df.columns = df.columns.str.strip()
    print(df.info()) 

    # Specify target column name
    target_column = cfg.target_col  # Replace with the name of the target variable column

    # Drop target column from the DataFrame
    df_features = df.drop(columns=[target_column])
    #Drop columns from the config file
    if cfg.preprocessing.drop_columns is not None:
        df_features = df_features.drop(columns=cfg.preprocessing.drop_columns)
    # Detect numerical and categorical columns
    numerical_columns = df_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df_features.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    #print how many missing values are there in each column
    print(df_features.isnull().sum())
    print("Numerical columns:", numerical_columns)
    print("Categorical columns:", categorical_columns)
    
    numerical_list = []
    category_list = []
    required_features = [name for name, required in cfg['features'].items() if required]
    print(required_features)
    # Construct the pipeline
    transformers = []

    if len(numerical_columns) > 0:
        # Check if there are missing values in the numerical columns
        if df_features[numerical_columns].isnull().any().any():
            numerical_list.append(("Num imputer", IterativeImputer(estimator=lgb.LGBMRegressor(), missing_values=np.nan, random_state=42)))
        for name in required_features:
        # Check if autofeature transform is required
            if name != 'scaler':
                if name == 'auto_feature':
                    auto_feature_transformer = AutoFeatureTransform(problem_type=cfg.problem_type)
                    numerical_list.append(("auto_feature", auto_feature_transformer))
                else:
                    numerical_list.append((name, FunctionTransformer(FEATURES_MAPPING[name])))
            # Check if scaler is required
            if name == 'scaler':
                numerical_list.append(("scaler", StandardScaler()))
        numerical_pipe = Pipeline(steps=numerical_list)
        transformers.append(('numeric', numerical_pipe, numerical_columns))

    if len(categorical_columns) > 0:
        category_list = [("Cat Imputer", SimpleImputer(strategy='most_frequent')), ("one-hot-ecoding", OneHotEncoder(handle_unknown='error',sparse_output =False))]
        categorical_pipe = Pipeline(steps=category_list)
        transformers.append(('categorical', categorical_pipe, categorical_columns))
    numerical_pipe = Pipeline(steps=numerical_list)
    categorical_pipe = Pipeline(steps=category_list)
    print(numerical_pipe)
    print(categorical_pipe)
    preprocessor = ColumnTransformer(transformers=transformers, verbose_feature_names_out=False).set_output(transform="pandas")

    print(preprocessor)
    


    print("X_train shape:", df_features.shape)
    print("y_train shape:", len(df[target_column]))
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df_features, df[target_column], test_size=cfg.preprocessing.test_size, random_state=cfg.random_state)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print(X_train.head())
    #Engineer the data
    X_train_engineered = preprocessor.fit_transform(X_train, y_train)
    print(X_train_engineered)
    X_test_engineered = preprocessor.transform(X_test)
    print(X_train_engineered.info())
    
    # ... code ...
    # Encode the target variable


    #Save the target transformer object
    #joblib.dump(preprocessing, 'preprocessing.joblib')
    # Save the target encoder object
    pipeline_path = os.path.join(model_dir, f'{cfg.competition_name}')
    os.makedirs(pipeline_path, exist_ok=True)
    joblib.dump(preprocessor, pipeline_path + "/pipeline.joblib")
    #print(full_pipeline.named_transformers_['feature_engineering'].named_transformers_['feature_eng'])
# Check if the parent directory exists, if not, create it
    holdout_dir = os.path.join(data_dir, f'processed/{cfg.competition_name}')
    os.makedirs(holdout_dir, exist_ok=True)
    holdout_path = os.path.join(data_dir, f'processed/{cfg.competition_name}/train_prepared.feather')
    X_train_engineered[target_column] = y_train.values
    X_train_engineered.reset_index(drop=True).to_feather(holdout_path)

    # Repeat the same for test data
    holdout_dir = os.path.join(data_dir, f'processed/{cfg.competition_name}')
    os.makedirs(holdout_dir, exist_ok=True)
    holdout_path = os.path.join(data_dir, f'processed/{cfg.competition_name}/test_holdout_prepared.feather')
    X_test_engineered[target_column] = y_test.values
    X_test_engineered.reset_index(drop=True).to_feather(holdout_path)
    
    print(X_train_engineered.reset_index(drop=True).info())
    print(X_test_engineered.reset_index(drop=True).info())
    print("The script is successfully completed")
    ############################################################"""


if __name__ == '__main__':
    main()
