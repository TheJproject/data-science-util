import joblib
import hydra
import logging
import os
import sys
import pandas as pd
import click
import wandb
import numpy as np
import joblib
import pyarrow.feather as feather
from sklearn.metrics import accuracy_score
from omegaconf import DictConfig
# Assuming that your_script.py and features_class.py are siblings in the directory structure
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../features'))
# Now the script should be able to import features_class
from features_class import FeatureNoTransform, CustomFeatureTransform, CustomTargetTransformer, CustomFeatureTransformFS,AutoFeatureTransform,ExternalFeatureTransform

# The rest of your script...

log = logging.getLogger(__name__)
#wandb.init()

@hydra.main(config_path="../..",config_name="config")
def main(cfg: DictConfig) -> None:
    working_dir = os.getcwd()
    data_dir = hydra.utils.to_absolute_path(cfg.data_dir)
    data_path = os.path.join(data_dir, 'cleaned/test.feather')
    model_dir = hydra.utils.to_absolute_path(cfg.model_dir)
    print(f"The current working directory for data is {data_dir}")
    print(f"The current working directory is {working_dir}")
    print(f"The current working directory for models is {model_dir}")
    log.info(f"Experiment: {working_dir}")

    if len(cfg.ensemble_list) > 1:
        model_path = os.path.join(model_dir, f'models_{cfg.selected_model}/stacking_model.pkl')
        model = joblib.load(model_path)
    else:
        model_path = os.path.join(model_dir, f'models_{cfg.selected_model}/{cfg.ensemble_list[0]}_model.pkl')
        model = joblib.load(model_path)

    #Load the feature_engineering pipeline for the test set 
    preprocessing_path = os.path.join(model_dir, f'models_{cfg.selected_model}/pipeline.joblib')
    feature_engineering = joblib.load(preprocessing_path)

    df = feather.read_feather(data_path)

    df.columns = df.columns.str.replace('\[.*?\]', '', regex=True)
    df.columns = df.columns.str.strip()
    
    print(df.head())
    print(df.info())
    print("\nBefore feature engineering, data shape: ", df.shape)

    df_engineered= feature_engineering.transform(df)
    print(df_engineered.info())
  
    print("\nAfter feature engineering, data shape: ", df_engineered.shape)
    # Make prediction on test data
    predictions = model.predict(df_engineered)
    print(predictions)#[:,1])
    # Create submission file
    submission_df = pd.DataFrame({'id': df.index, cfg.target_col: predictions})#[:,1]})
    data_path = os.path.join(data_dir, 'submission/submission.csv')
    submission_df.to_csv(data_path, index=False)


if __name__ == "__main__":
    main()
