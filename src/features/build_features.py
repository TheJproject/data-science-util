from features_class import FeatureNoTransform, CustomFeatureTransform, CustomTargetTransformer, CustomFeatureTransformFS,AutoFeatureTransform,ExternalFeatureTransform
import click
import logging
import pyarrow.feather as feather
import pandas as pd
import joblib
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@hydra.main(config_path="../..",config_name="config")
def main(cfg: DictConfig) -> None:

    # Handling directories and paths
    working_dir = os.getcwd()
    data_dir = hydra.utils.to_absolute_path(cfg.data_dir)
    data_path = os.path.join(data_dir, 'cleaned/train.feather')
    model_dir = hydra.utils.to_absolute_path(cfg.model_dir)
    print(f"The current working directory for data is {data_dir}")
    print(f"The current working directory is {working_dir}")
    print(f"The current working directory for models is {model_dir}")
    log.info(f"Experiment: {working_dir}")

    # Load data
    df = feather.read_feather(data_path)

    # check the data is loaded correctly
    print(df.head()) 

    # Specify target column name
    target_column = cfg.target_col  # Replace with the name of the target variable column

    # Drop target column from the DataFrame
    df_features = df.drop(columns=[target_column])

    # Detect numerical and categorical columns
    numerical_columns = df_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df_features.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

    print("Numerical columns:", numerical_columns)
    print("Categorical columns:", categorical_columns)

    feature_engineering_list = []
    preprocessor_list = []

    if cfg.features.no_feature == True:
        feature_engineering_list.append(("no_features", FeatureNoTransform(), numerical_columns))
    if cfg.features.custom_feature == True:
        feature_engineering_list.append(("custom_features", CustomFeatureTransform(), numerical_columns))
    if cfg.features.auto_feature == True:
        feature_engineering_list.append(("auto_features", AutoFeatureTransform(**cfg), numerical_columns))
        auto_features = ("auto_features", AutoFeatureTransform(**cfg))
    if cfg.features.ext_feature == True:
        feature_engineering_list.append(("ext_features", ExternalFeatureTransform(), numerical_columns))

    #if len(numerical_columns) > 0:
    #    preprocessor_list.append(("scaler", StandardScaler(), numerical_columns))
    #if len(categorical_columns) > 0:
    #    preprocessor_list.append(("one-hot-ecoding", OneHotEncoder(), categorical_columns))
    print("Feature list " + str(feature_engineering_list))

    """Test FeatureUnion """
    numeric_features = ("numeric_features", 'passthrough')
    # Use FeatureUnion to combine the features
    feature_union = FeatureUnion(transformer_list=[numeric_features, auto_features])

    feature_engineering = ColumnTransformer(transformers=[('features', feature_union, numerical_columns)]) #feature_engineering_list
    #preprocessing = ColumnTransformer(transformers=preprocessor_list)

    # Combine preprocessing with custom feature engineering
    #full_pipeline = Pipeline(steps=[
    #        ("feature_engineering", feature_engineering),
    #        ("preprocessor", preprocessing),
    #    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df_features, df[target_column], test_size=cfg.preprocessing.test_size, random_state=cfg.random_state)
    print(y_train.isna().sum())
    #Engineer the data
    X_train_engineered = feature_engineering.fit_transform(X_train, y_train)
    print(X_train_engineered)
    X_test_engineered = feature_engineering.transform(X_test)


    # Get transformed column names from AutoFeatureTransform
    if cfg.features.auto_feature and 'auto_features' in feature_engineering.named_transformers_:
        transformed_numerical_columns = feature_engineering.named_transformers_['auto_features'].get_transformed_columns()
    else:
        transformed_numerical_columns = numerical_columns


    # Update preprocessor_list with new column names
    preprocessor_list = []
    if len(transformed_numerical_columns) > 0:
        preprocessor_list.append(("scaler", StandardScaler(), list(range(len(transformed_numerical_columns)))))
    if len(categorical_columns) > 0:
        preprocessor_list.append(("one-hot-encoding", OneHotEncoder(), list(range(len(transformed_numerical_columns), len(transformed_numerical_columns) + len(categorical_columns)))))

    preprocessing = ColumnTransformer(transformers=preprocessor_list)

    # Preprocess the data
    X_train_prepared = preprocessing.fit_transform(X_train_engineered)
    print("shape of X_train_prepared: " + str(X_train_prepared.shape))
    X_test_prepared = preprocessing.transform(X_test_engineered)

    # Encode the target variable
    #target_encoder = CustomTargetTransformer()
    #y_train_encoded = target_encoder.fit_transform(y_train)
    #y_test_encoded = target_encoder.transform(y_test)

    #Save the target transformer object
    joblib.dump(preprocessing, 'preprocessing.joblib')
    # Save the target encoder object
    joblib.dump(feature_engineering, "feature_engineering.joblib")
    #print(full_pipeline.named_transformers_['feature_engineering'].named_transformers_['feature_eng'])

    #custom_feature_names = full_pipeline.named_transformers_['feature_engineering'].named_transformers_["feature_eng"].get_feature_names_out()
    #print("len feature name: " + str(len(custom_feature_names)))
    #print(custom_feature_names)
    #column_names = numerical_columns + custom_feature_names
    #print("shape of X_train_prepared: " + str(X_train_prepared.shape))
    train_df_prepared = pd.DataFrame(X_train_prepared, columns=transformed_numerical_columns)
    print(y_train.isna().sum()) 
    train_df_prepared[target_column] = y_train.values
    #train_df_prepared.reset_index(drop=True, inplace=True)
    test_df_prepared = pd.DataFrame(X_test_prepared, columns=transformed_numerical_columns)
    test_df_prepared[target_column] = y_test.values
    #test_df_prepared.reset_index(drop=True, inplace=True)
    # Save the preprocessed train and testidation sets as .feather files
    holdout_path = os.path.join(data_dir, 'processed/train_prepared.feather')
    train_df_prepared.to_feather(holdout_path)
    holdout_path = os.path.join(data_dir, 'processed/test_holdout_prepared.feather')
    test_df_prepared.to_feather(holdout_path)
    print("The script is successfully completed")
    ############################################################"""


if __name__ == '__main__':
    main()
