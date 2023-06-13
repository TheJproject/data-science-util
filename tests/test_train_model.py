import pytest
import pandas as pd
import os
import joblib
#from  src.models.train_model import main 
import hydra
from hydra.experimental import compose, initialize
#
@pytest.fixture
def cfg():
    with initialize(config_path="../.."):
        cfg = compose(config_name="config")
    return cfg

@pytest.fixture
def train_df_prepared():
    # Prepare a sample DataFrame
    data = {
        'num_0': [0.1, 0.2, 0.3, 0.4, 0.5],
        'num_1': [0.5, 0.4, 0.3, 0.2, 0.1],
        'cat_0': ['A', 'B', 'C', 'D', 'E'],
        'target': [1, 0, 0, 1, 0]
    }
    train_df_prepared = pd.DataFrame(data)
    return train_df_prepared

@pytest.fixture
def mock_feather(mocker, train_df_prepared):
    # Mock the feather.read_feather function to return the sample DataFrame
    mocker.patch('feather.read_feather', return_value=train_df_prepared)

"""def test_main(cfg, mock_feather):
    # Run the main function
    main(cfg)

    # Assert that model and parameters were saved correctly
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    for model_name in cfg.ensemble_list:
        assert os.path.exists(f"{cfg.model_dir}/models_{timestamp}/{model_name}_param.pkl")
        assert os.path.exists(f"{cfg.model_dir}/models_{timestamp}/{model_name}_model.pkl")

        params = joblib.load(f"{cfg.model_dir}/models_{timestamp}/{model_name}_param.pkl")
        assert isinstance(params, dict)

        model = joblib.load(f"{cfg.model_dir}/models_{timestamp}/{model_name}_model.pkl")
        # Check the type of the model depending on the name
        # This is just an example, update with the correct types
        if model_name == 'RF':
            assert isinstance(model, RandomForestClassifier)
        elif model_name == 'CatBoost':
            # Check the type for CatBoost
            pass
        # Add more elif statements for the other models
"""