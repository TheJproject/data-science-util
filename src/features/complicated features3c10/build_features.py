from features_class import FeatureNoTransform, CustomFeatureTransform, CustomTargetTransformer, CustomFeatureTransformFS
import click
import logging
import pyarrow.feather as feather
import pandas as pd
import joblib
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

class renamer():
    def __init__(self):
        self.d = dict()

    def __call__(self, x):
        if x not in self.d:
            self.d[x] = 0
            return x
        else:
            self.d[x] += 1
            return "%s_%d" % (x, self.d[x])

@hydra.main(config_path="../..",config_name="config")
def main(cfg: DictConfig) -> None:
    working_dir = os.getcwd()
    data_dir = hydra.utils.to_absolute_path(cfg.data_dir)
    data_path = os.path.join(data_dir, 'cleaned/data.feather')

    model_dir = hydra.utils.to_absolute_path(cfg.model_dir)
    print(f"The current working directory for data is {data_dir}")
    print(f"The current working directory is {working_dir}")
    print(f"The current working directory for models is {model_dir}")
    log.info(f"Experiment: {working_dir}")

    # Load data
    df = feather.read_feather(data_path)
    # Specify target column name
    target_column = cfg.target_col  # Replace with the name of the target variable column

    # Drop target column from the DataFrame
    df_features = df.drop(columns=[target_column])

    # Detect numerical and categorical columns
    numerical_columns = df_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df_features.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

    print("Numerical columns:", numerical_columns)
    print("Categorical columns:", categorical_columns)

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            ("custom_features", CustomFeatureTransformFS(), numerical_columns),
            ("cat", OneHotEncoder(), categorical_columns),
        ]
    )

    # Combine preprocessing with custom feature engineering
    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("custom_features", CustomFeatureTransformFS()),
        ]
    )

    # Train-test split
    train_set, test_set = train_test_split(df, test_size=cfg.preprocessing.test_size, random_state=cfg.random_state)

    # Separate target variable from features
    X_train = train_set[numerical_columns]
    y_train = train_set[target_column]
    X_test = test_set[numerical_columns]
    y_test = test_set[target_column]
    # Preprocess the data
    X_train_prepared = preprocessor.fit_transform(X_train)
    X_test_prepared = preprocessor.transform(X_test)

    # Encode the target variable
    target_encoder = CustomTargetTransformer()
    y_train_encoded = target_encoder.fit_transform(y_train)
    y_test_encoded = target_encoder.transform(y_test)

    # Save the target transformer object
    joblib.dump(preprocessor, 'full_pipeline.joblib')
    # Save the target encoder object
    joblib.dump(target_encoder, "target_encoder.joblib")

    # Re-create DataFrames with preprocessed data and combine with encoded target variable
    if categorical_columns != []:
        column_names = numerical_columns + list(preprocessor.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_columns))
    else:
        column_names = numerical_columns

    new_feature_names = preprocessor.named_transformers_["custom_features"].get_new_feature_names()
    print("len feature name: " + str(len(new_feature_names)))

    column_names = numerical_columns + list(preprocessor.named_transformers_["custom_features"].get_new_feature_names())
    train_df_prepared = pd.DataFrame(X_train_prepared, columns=column_names)
    train_df_prepared[target_column] = y_train_encoded
    train_df_prepared.reset_index(drop=True, inplace=True)
    test_df_prepared = pd.DataFrame(X_test_prepared, columns=column_names)
    test_df_prepared[target_column] = y_test_encoded
    test_df_prepared.reset_index(drop=True, inplace=True)
    print(test_df_prepared.columns)
    num_duplicate_cols = train_df_prepared.columns.duplicated().sum()
    print(f'The DataFrame has {num_duplicate_cols} duplicate column name(s).')
    train_df_prepared=train_df_prepared.rename(columns=renamer())
    # Save the preprocessed train and testidation sets as .feather files
    holdout_path = os.path.join(data_dir, 'processed/train_prepared.feather')
    train_df_prepared.to_feather(holdout_path)
    holdout_path = os.path.join(data_dir, 'processed/test_holdout_prepared.feather')
    test_df_prepared.to_feather(holdout_path)
    print("The script is successfully completed")
    ############################################################


if __name__ == '__main__':
    main()
