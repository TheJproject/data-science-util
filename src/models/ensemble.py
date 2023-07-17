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
from itertools import chain, combinations

from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score
from sklearn.metrics import accuracy_score, make_scorer, log_loss, mean_absolute_error, get_scorer
from sklearn.ensemble import StackingClassifier,VotingClassifier,VotingRegressor

from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import SCORERS
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import torch
import objectives as obj

log = logging.getLogger(__name__)
#wandb.init()

class EnsembleModel(BaseEstimator, ClassifierMixin):
    def __init__(self, models, is_pytorch):
        self.models = models
        self.is_pytorch = is_pytorch

    def predict_proba(self, X):
        # Calculate the ensemble prediction
        predictions = []
        for model, is_pytorch in zip(self.models, self.is_pytorch):
            if is_pytorch:
                X_tensor = torch.tensor(X.values, dtype=torch.float32)
                with torch.no_grad():
                    prediction = model(X_tensor).numpy()
            else:
                prediction = model.predict_proba(X)
            predictions.append(prediction)

        return np.mean(predictions, axis=0)

@hydra.main(config_path="../..",config_name="config")
def main(cfg: DictConfig) -> None:
    working_dir = os.getcwd()
    data_dir = hydra.utils.to_absolute_path(cfg.data_dir)
    data_path = os.path.join(data_dir, f'processed/{cfg.competition_name}/data.feather')

    train_prepared_path = os.path.join(data_dir, f'processed/{cfg.competition_name}/train_prepared.feather')
    train_df_prepared = feather.read_feather(train_prepared_path)
    test_holdout_path = os.path.join(data_dir, f'processed/{cfg.competition_name}/test_holdout_prepared.feather')
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
        if name == 'NN' or name == 'Autoencoder':
            model_path = os.path.join(model_dir, f'{cfg.competition_name}/models_{cfg.selected_model}/{name}_model.pt')
            model = torch.load(model_path)
            base_estimators.append((name, model, True))
        else:
            model_path = os.path.join(model_dir, f'{cfg.competition_name}/models_{cfg.selected_model}/{name}_model.pkl')
            model = joblib.load(model_path)
            base_estimators.append((name, model, False))



    # Metric handling
    if cfg.metric in SCORERS:
        metric = SCORERS[cfg.metric]._score_func
    elif cfg.metric in globals() and callable(globals()[cfg.metric]):
        metric = globals()[cfg.metric]
    else:
        raise ValueError(f"Unsupported metric: {cfg.metric}")


    # Placeholder for results
    all_res = {}
    best_estimators = None
    if cfg.direction == 'minimize':
        best_score = np.inf  # lower is better
    elif cfg.direction == 'maximize':
        best_score = -np.inf  # higher is better
    else:
        raise ValueError(f"Unsupported direction: {cfg.direction}, choose either 'minimize' or 'maximize'")

    def powerset(iterable):
        # This function returns all subsets of a set (including the empty set and the set itself)
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


    # Iterate over each possible combination of models
    for subset in powerset(base_estimators):
        if len(subset) == 0:
            continue

        # Name of the current ensemble
        names = [model[0] for model in subset]
        name_combo = "+".join(names)
        
        # Get the ensemble prediction
        predictions = []

        for model in subset:
            name, estimator, is_pytorch = model  # Unpack the tuple
            if is_pytorch:  # Check if the model is a PyTorch model
                # PyTorch model
                X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
                with torch.no_grad():
                    output = estimator.forward(X_test_tensor)
                    softmax_output = torch.nn.functional.softmax(output, dim=1)
                    prediction = softmax_output.detach().numpy()[:, 1]
            else:
                # Scikit-learn model (including XGBClassifier)
                prediction = estimator.predict_proba(X_test)[:, 1]
            predictions.append(prediction)



        y_pred_combo = np.mean(predictions, axis=0)


        # Compute score
        score = round(metric(y_test, y_pred_combo), 5)

        # Store the results
        all_res[name_combo] = score

        # Check if it's the best score
        if (cfg.direction == 'minimize' and score < best_score) or (cfg.direction == 'maximize' and score > best_score):
            best_score = score
            best_estimators = subset


        # Log the results
        log.info(f'Ensemble: {name_combo}, Score: {score}')

    # Log the best result
    log.info(f'Best ensemble: {[model[0] for model in best_estimators]}, Score: {best_score}')

    best_ensemble = EnsembleModel([model[1] for model in best_estimators], [model[2] for model in best_estimators])


    model_path = os.path.join(model_dir, f'{cfg.competition_name}/models_{cfg.selected_model}/stacking_model.pkl')
    joblib.dump(best_ensemble, model_path)

    

if __name__ == "__main__":
    main()
