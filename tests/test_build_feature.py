import pytest
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder
#from imblearn.over_sampling import SMOTE
from pandas.api.types import is_numeric_dtype
from unittest.mock import patch, Mock
from omegaconf import DictConfig
import hydra
from hydra.experimental import compose, initialize
import os
from tests import _PATH_DATA
from src.data import make_dataset  
from src.features import build_features2  
from src.features.features_class import (  # replace 'your_module' with the actual name of your module
    FeatureNoTransform,
    AutoFeatureTransform,
    ExternalFeatureTemplate,
    CustomFeatureTransform,
    CustomFeatureTransformFS,
    CustomTargetTransformer,
    FEATURES_MAPPING
)
#
@pytest.fixture(scope="session")
def cfg():
    with initialize(config_path="../.."):
        cfg = compose(config_name="config")
    return cfg

@pytest.fixture
def dummy_dataframe():
    return pd.DataFrame({
        'NumericalCol1': [1, 2, 3],
        'NumericalCol2': [4, 5, 6],
        'CategoricalCol1': ['a', 'b', 'c'],
        'CategoricalCol2': ['d', 'e', 'f'],
    })


def test_main(dummy_dataframe):
    # Test whether the data is loaded correctly
    @patch('build_features2.feather.read_feather')
    def test_load_data(mock_read_feather):
        # Arrange
        mock_read_feather.return_value = dummy_dataframe
        expected_path = os.path.join(_PATH_DATA, 'cleaned/train.feather')

        # Act
        build_features2.main(Mock())  

        # Assert
        mock_read_feather.assert_called_once_with(expected_path)

    # Test renaming of columns
    @patch('build_features2.feather.read_feather')
    def test_rename_columns(mock_read_feather):
        # Arrange
        df = dummy_dataframe.copy()
        df.columns = ["NumericalCol1[units]", " NumericalCol2 ", "CategoricalCol1[units]", "CategoricalCol2[units]"]
        mock_read_feather.return_value = df

        # Act
        build_features2.main(Mock())  

        # Assert
        expected_columns = ["NumericalCol1", "NumericalCol2", "CategoricalCol1", "CategoricalCol2"]
        pd.testing.assert_frame_equal(df.columns, expected_columns)

    # Test train_test_split function call
    @patch('build_features2.train_test_split')
    def test_train_test_split(mock_train_test_split):
        # Arrange
        mock_train_test_split.return_value = [pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()]

        # Act
        build_features2.main(Mock())  

        # Assert
        assert mock_train_test_split.called

    # Add more tests as needed for other parts of your function
def generate_test_data(m, n):
    numerical_data = np.random.rand(100, m)
    categorical_data = np.random.choice(['A', 'B', 'C'], size=(100, n))

    # Create separate dataframes for numerical and categorical data
    num_df = pd.DataFrame(numerical_data, columns=[f'num_{i}' for i in range(m)])
    cat_df = pd.DataFrame(categorical_data, columns=[f'cat_{i}' for i in range(n)])

    # Concatenate dataframes along columns
    data = pd.concat([num_df, cat_df], axis=1)

    data['x_e_out [-]'] = np.random.randint(0, 2, 100)  # Target column

    return data

@pytest.fixture
def data():
    return generate_test_data(5, 2)  # assuming 5 numerical columns and 2 categorical ones

# Test for the FeatureNoTransform class
def test_feature_no_transform(data):
    transformer = FeatureNoTransform(data)
    transformed_data = transformer.transform()
    pd.testing.assert_frame_equal(transformed_data, data)

# Test for the AutoFeatureTransform class
def test_auto_feature_transform(data):
    # Encode categorical columns
    data.columns = data.columns.astype(str)
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    numerical_data = data.select_dtypes(include=[np.number])
    categorical_data = data.select_dtypes(exclude=[np.number])
    
    encoded_categorical_data = encoder.fit_transform(categorical_data)
    encoded_categorical_df = pd.DataFrame(encoded_categorical_data, 
                                          columns=[f'cat_encoded_{i}' for i in range(encoded_categorical_data.shape[1])])
    
    data_encoded = pd.concat([numerical_data.reset_index(drop=True), encoded_categorical_df], axis=1)

    transformer = AutoFeatureTransform('regression')
    transformer.fit(data_encoded, data['x_e_out [-]'])


def test_auto_feature_transform_not_fitted(data):
    transformer = AutoFeatureTransform('regression')
    with pytest.raises(NotFittedError):
        transformer.transform(data)

# Test for the ExternalFeatureTemplate class
#def test_external_feature_template(data):
#    transformer = ExternalFeatureTemplate()
#    transformer.fit(data)
#    assert transformer.indices is not None
#    assert transformer.column_names_ is not None

# Test for the CustomFeatureTransform class
#def test_custom_feature_transform(data):
#    transformer = CustomFeatureTransform(data)
#    transformer.fit(data, data['x_e_out [-]'])
#    assert transformer.smote is not None

# Test for the CustomFeatureTransformFS class
def test_custom_feature_transform_fs(data):
    transformer = CustomFeatureTransformFS()
    transformer.fit(data)
    assert transformer.FEATS is not None

# Test for the CustomTargetTransformer class
def test_custom_target_transformer(data):
    transformer = CustomTargetTransformer()
    transformer.fit(data['x_e_out [-]'])
    assert transformer.encoder is not None
    assert transformer.is_classification is not None

# Test for the FEATURES_MAPPING dictionary
def test_features_mapping():
    assert set(FEATURES_MAPPING.keys()) == {"no_feature", "custom_feature", "auto_feature", "ext_feature"}

if __name__ == "__main__":
    pytest.main()

