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

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score
from sklearn.metrics import accuracy_score, cohen_kappa_score

from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)
wandb.init()

@hydra.main(config_path="../..",config_name="config")
def main(cfg: DictConfig) -> None:
    working_dir = os.getcwd()
    data_dir = hydra.utils.to_absolute_path(cfg.data_dir)
    data_path = os.path.join(data_dir, 'processed/train_prepared.feather')

    model_dir = hydra.utils.to_absolute_path(cfg.model_dir)
    print(f"The current working directory for data is {data_dir}")
    print(f"The current working directory is {working_dir}")
    print(f"The current working directory for models is {model_dir}")
    log.info(f"Experiment: {working_dir}")

    # Load data
    train_df_prepared = feather.read_feather(data_path)
    # Specify target column name
    target_column = cfg.target_col  # Replace with the name of the target variable column

    # Split the DataFrames into feature matrices (X) and target vectors (y)
    X_train = train_df_prepared.drop(columns=[target_column])
    y_train = train_df_prepared[target_column]
    #X.reset_index(drop=True, inplace=True)
    #y.reset_index(drop=True, inplace=True)
    # Split the data into train and holdout sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.preprocessing.test_size, random_state=cfg.random_state)

    # Split data into training and testing sets
    kfold = StratifiedKFold(n_splits=cfg.preprocessing.nsplit,shuffle=True,random_state=cfg.random_state)
    if cfg.OPTUNA == True:
    
        # Dictionary containing classification and regression models
        model_dict = {
            "classification": {
                "RF": obj.ObjectiveRF,
                "CatBoost": obj.ObjectiveCatBoost,
                "XGBoost": obj.ObjectiveXGB,
                "LGBM": obj.ObjectiveLGBM,
            },
            "regression": {
                "RF": obj.ObjectiveRFRegressor,
                "CatBoost": obj.ObjectiveCatBoostRegressor,
                "XGBoost": obj.ObjectiveXGBRegressor,
                "LGBM": obj.ObjectiveLGBMRegressor,
            },
        }

        # Filter the dictionary based on cfg.problem_type and cfg.ensemble_list
        models = {
            k: v(kfold, X_train, y_train)
            for k, v in model_dict[cfg.problem_type].items()
            if k in cfg.ensemble_list
        }
        
        log.info(f"Ensemble list: {cfg.ensemble_list}")
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        #base_preds_test = np.empty((X_test.shape[0], len(models)))

        for i, (name, model) in enumerate(models.items()):
            study = obj.MyOptimizer(model)
            study.optimize(n_trials=cfg.hypopt.n_trials)
            best_param = study.best_params()
            best_model = model.best_model(best_param)
            print(best_param)
            print(study.best_value())
            #base_preds_train[:, i] = cross_val_predict(model, X_train, y_train, cv=kfold, method='predict_proba')[:, 1]
            #base_preds_test[:, i] = best_model.predict(X_test)[:, 1]
            log.info(f"{name} Parameters: {best_param}")
            os.makedirs(f"{model_dir}/models_{timestamp}" , exist_ok=True)
            param_path = os.path.join(model_dir, f'models_{timestamp}/{name}_param.pkl')
            joblib.dump(best_param, param_path)
            model_path = os.path.join(model_dir, f'models_{timestamp}/{name}_model.pkl')
            joblib.dump(best_model, model_path)
            


    

if __name__ == "__main__":
    main()
