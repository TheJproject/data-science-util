import hydra
import logging
import os
import pandas as pd
import click
import wandb
import numpy as np
import datetime
import joblib
import pyarrow.feather as feather
import objectives as obj
import pickle
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score
from sklearn.metrics import accuracy_score

from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)
wandb.init()


class Ensemble:
    def __init__(self, cfg: DictConfig):
        self.config = cfg

    def train(self):
        num_epochs = self.config.training.num_epochs
        learning_rate = self.config.training.learning_rate

        # Train the model
        # ...

    def evaluate(self):
        batch_size = self.config.testing.batch_size

@hydra.main(config_path="../..",config_name="config")
def main(cfg: DictConfig) -> None:
    working_dir = os.getcwd()
    data_dir = hydra.utils.to_absolute_path(cfg.data_dir)
    holdout_path = os.path.join(data_dir, 'processed/test_holdout.feather')

    model_dir = hydra.utils.to_absolute_path(cfg.model_dir)
    print(f"The current working directory for data is {data_dir}")
    print(f"The current working directory is {working_dir}")
    print(f"The current working directory for models is {model_dir}")
    df = pd.read_feather(holdout_path)
    y_test = df[cfg.target_col]
    X_test = df.drop(columns=[cfg.target_col])
    base_preds_test = np.empty((X_test.shape[0], len(cfg.ensemble_list)))
    for i, name in enumerate(cfg.ensemble_list):
        model_path = os.path.join(model_dir, f'models_{cfg.selected_model}/{name}_model.pkl')
        model = joblib.load(model_path)
        print(base_preds_test[:, i])
        base_preds_test[:, i] = model.predict(X_test)
    print(base_preds_test)
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(base_preds_test, y_test)


if __name__ == "__main__":
    main()