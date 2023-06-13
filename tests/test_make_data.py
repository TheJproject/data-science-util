import pytest
import pandas as pd
from unittest.mock import patch, Mock
import os
from tests import _PATH_DATA
from src.features import features_class #import FeatureNoTransform, CustomFeatureTransform, CustomTargetTransformer, CustomFeatureTransformFS, AutoFeatureTransform,ExternalFeatureTransform
from src.data import make_dataset  # replace this with the actual name of your script
from src.features import build_features2  # replace this with the actual name of your script


def test_main():
    # Test whether the data is loaded correctly
    @patch('make_dataset.feather.read_feather')
    def test_load_data(mock_read_feather):
        # Arrange
        mock_read_feather.return_value = pd.DataFrame()
        expected_path = os.path.join(_PATH_DATA, 'cleaned/train.feather')

        # Act
        make_dataset.main(Mock())  # using a Mock object as the cfg parameter

        # Assert
        mock_read_feather.assert_called_once_with(expected_path)

    # Test renaming of columns
    @patch('make_dataset.feather.read_feather')
    def test_rename_columns(mock_read_feather):
        # Arrange
        df = pd.DataFrame(columns=["col1[units]", " col2 ", "col3[units]"])
        mock_read_feather.return_value = df

        # Act
        make_dataset.main(Mock())  # using a Mock object as the cfg parameter

        # Assert
        expected_columns = ["col1", "col2", "col3"]
        pd.testing.assert_frame_equal(df.columns, expected_columns)

    # Test train_test_split function call
    @patch('make_dataset.train_test_split')
    def test_train_test_split(mock_train_test_split):
        # Arrange
        mock_train_test_split.return_value = [pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()]

        # Act
        make_dataset.main(Mock())  # using a Mock object as the cfg parameter

        # Assert
        assert mock_train_test_split.called

    # Add more tests as needed for other parts of your function


if __name__ == "__main__":
    pytest.main()
