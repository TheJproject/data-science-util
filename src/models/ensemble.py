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
from sklearn.metrics import accuracy_score, make_scorer, log_loss, mean_absolute_error, get_scorer
from sklearn.ensemble import StackingClassifier,VotingClassifier,VotingRegressor

from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)
wandb.init()

@hydra.main(config_path="../..",config_name="config")
def main(cfg: DictConfig) -> None:
    working_dir = os.getcwd()
    data_dir = hydra.utils.to_absolute_path(cfg.data_dir)
    data_path = os.path.join(data_dir, 'processed/data.feather')

    train_prepared_path = os.path.join(data_dir, 'processed/train_prepared.feather')
    train_df_prepared = feather.read_feather(train_prepared_path)
    test_holdout_path = os.path.join(data_dir, 'processed/test_holdout_prepared.feather')
    test_df_prepared = feather.read_feather(test_holdout_path)
    model_dir = hydra.utils.to_absolute_path(cfg.model_dir)

    print(train_df_prepared.info())
    print(f"The current working directory for data is {data_dir}")
    print(f"The current working directory is {working_dir}")
    print(f"The current working directory for models is {model_dir}")
    log.info(f"Experiment: {working_dir}")

    # Load data
    X_train = train_df_prepared.drop(columns=[cfg.target_col])
    X_test = test_df_prepared.drop(columns=[cfg.target_col])
    y_train = train_df_prepared[cfg.target_col]
    y_test = test_df_prepared[cfg.target_col]

    print(X_train.info())

    base_estimators = []
    
    for name in cfg.ensemble_list:
        param_path = os.path.join(model_dir, f'models_{cfg.selected_model}/{name}_param.pkl')
        best_param = joblib.load(param_path)
        model_path = os.path.join(model_dir, f'models_{cfg.selected_model}/{name}_model.pkl')
        best_model = joblib.load(model_path)
        base_estimators.append((name, best_model))

    # Define the meta-classifier
    meta_classifier = LogisticRegression()

    # Create the stacking classifier
    stacking_classifier = VotingRegressor(estimators=base_estimators)#, voting='soft')#final_estimator=meta_classifier, verbose=1)

    # Train the stacking classifier
    stacking_classifier.fit(X_train, y_train)
    model_path = os.path.join(model_dir, f'models_{cfg.selected_model}/stacking_model.pkl')
    joblib.dump(stacking_classifier, model_path)

    # Test the stacking classifier
    if cfg.metric == 'rmse':
        scorer = make_scorer(obj.root_mean_squared_error, greater_is_better=False)
    else:
        scorer = get_scorer(cfg.metric)
    score = scorer(stacking_classifier, X_test, y_test)
    accuracy = stacking_classifier.score(X_test, y_test)
    print(f'Stacking classifier accuracy: {score}')

    

if __name__ == "__main__":
    main()
