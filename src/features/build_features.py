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

    # Preprocessing ColumnTransformer
    feature_engineering = ColumnTransformer(
        transformers=[
            ("feature_eng", CustomFeatureTransformFS(),numerical_columns)
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
        ]
    )

    # Combine preprocessing with custom feature engineering
    full_pipeline = ColumnTransformer(
        transformers=[
            ("feature_engineering", feature_engineering,numerical_columns),
            ("preprocessor", preprocessor,numerical_columns),
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
    X_train_prepared = full_pipeline.fit_transform(X_train)
    print("shape of X_train_prepared: " + str(X_train_prepared.shape))
    X_test_prepared = full_pipeline.transform(X_test)

    # Encode the target variable
    target_encoder = CustomTargetTransformer()
    y_train_encoded = target_encoder.fit_transform(y_train)
    y_test_encoded = target_encoder.transform(y_test)

    # Save the target transformer object
    joblib.dump(full_pipeline, 'full_pipeline.joblib')
    # Save the target encoder object
    joblib.dump(target_encoder, "target_encoder.joblib")
    print(full_pipeline.named_transformers_['feature_engineering'].named_transformers_['feature_eng'])

    custom_feature_names = full_pipeline.named_transformers_['feature_engineering'].named_transformers_["feature_eng"].get_feature_names_out()
    print("len feature name: " + str(len(custom_feature_names)))
    print(custom_feature_names)
    column_names = numerical_columns + custom_feature_names
    print("shape of X_train_prepared: " + str(X_train_prepared.shape))
    train_df_prepared = pd.DataFrame(X_train_prepared, columns=column_names)
    train_df_prepared[target_column] = y_train_encoded
    train_df_prepared.reset_index(drop=True, inplace=True)
    test_df_prepared = pd.DataFrame(X_test_prepared, columns=column_names)
    test_df_prepared[target_column] = y_test_encoded
    test_df_prepared.reset_index(drop=True, inplace=True)
    # Save the preprocessed train and testidation sets as .feather files
    holdout_path = os.path.join(data_dir, 'processed/train_prepared.feather')
    train_df_prepared.to_feather(holdout_path)
    holdout_path = os.path.join(data_dir, 'processed/test_holdout_prepared.feather')
    test_df_prepared.to_feather(holdout_path)
    print("The script is successfully completed")
    ############################################################


if __name__ == '__main__':
    main()
