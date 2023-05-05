import joblib
import hydra
import logging
import os
import pandas as pd
import click
import wandb
import numpy as np
import joblib
import pyarrow.feather as feather
from sklearn.metrics import accuracy_score
from omegaconf import DictConfig

log = logging.getLogger(__name__)
wandb.init()

@hydra.main(config_path="../..",config_name="config")
def main(cfg: DictConfig) -> None:
    working_dir = os.getcwd()
    data_dir = hydra.utils.to_absolute_path(cfg.data_dir)
    data_path = os.path.join(data_dir, 'processed/test.feather')
    model_dir = hydra.utils.to_absolute_path(cfg.model_dir)
    print(f"The current working directory for data is {data_dir}")
    print(f"The current working directory is {working_dir}")
    print(f"The current working directory for models is {model_dir}")
    log.info(f"Experiment: {working_dir}")


    model_path = os.path.join(model_dir, f'models_{cfg.selected_model}/stacking_model.pkl')
    model = joblib.load(model_path)
    
    df = feather.read_feather(data_path)
    print(df.head())    
    # Make prediction on test data
    predictions = model.predict(df)
    print(predictions)#[:,1])
    # Create submission file
    submission_df = pd.DataFrame({'id': df.index, cfg.target_col: predictions})#[:,1]})
    data_path = os.path.join(data_dir, 'submission/submission.csv')
    submission_df.to_csv(data_path, index=False)


if __name__ == "__main__":
    main()
