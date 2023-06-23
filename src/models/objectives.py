import xgboost as xgb
import lightgbm as lgb
import catboost
from sklearn.metrics import accuracy_score, cohen_kappa_score,log_loss, get_scorer, make_scorer,mean_squared_error
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso
import xgboost as xgb
import lightgbm as lgb
import hydra
import optuna
import numpy as np
from optuna.integration.wandb import WeightsAndBiasesCallback
from sklearn.metrics import SCORERS

from omegaconf import DictConfig
#Additional Scorer

def rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred,squared=False)
    return -mse

# Classifier
class ObjectiveRF:
    def __init__(self, kfold, X, y, metric, random_state):
        self.kfold = kfold
        self.X = X
        self.y = y
        if metric in SCORERS:
            self.scorer = get_scorer(metric)
        elif metric in globals() and callable(globals()[metric]):
            self.scorer = make_scorer(globals()[metric], greater_is_better=False)
        else:
            # Here you can add more custom scorers if you need
            raise ValueError(f"Unsupported metric: {metric}")
        self.random_state = random_state
        
    def __call__(self, trial):        
        params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': self.random_state
    }
        optuna_model = RandomForestClassifier(**params)
        scores = []

        for train_index, val_index in self.kfold.split(self.X,self.y):
            # Split data into training and validation sets
            X_train, y_train = self.X.iloc[train_index], self.y[train_index]
            X_val, y_val = self.X.iloc[val_index], self.y[val_index]
            

            # Train model
            optuna_model.fit(X_train,y_train)
            score = self.scorer(optuna_model, X_val, y_val)
            print(score)
            scores.append(score)

        return score
        
    def best_model(self, best_param):
        best_model = RandomForestClassifier(**best_param)
        best_model.fit(self.X,self.y)
        return best_model

class ObjectiveCatBoost:
    def __init__(self, kfold, X, y, metric, random_state):
        self.kfold = kfold
        self.X = X
        self.y = y
        if metric in SCORERS:
            self.scorer = get_scorer(metric)
        elif metric in globals() and callable(globals()[metric]):
            self.scorer = make_scorer(globals()[metric], greater_is_better=False)
        else:
            # Here you can add more custom scorers if you need
            raise ValueError(f"Unsupported metric: {metric}")
        self.random_state = random_state

    def __call__(self, trial):
        params = {
            'task_type': 'CPU',
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'iterations': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1, log=True),
            'depth': trial.suggest_int('depth', 1, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 100, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'border_count': trial.suggest_int('border_count', 1, 255),
            'verbose': 0,
            'random_seed': 42
        }
        optuna_model = catboost.CatBoostClassifier(**params)
        scores = []

        for train_index, val_index in self.kfold.split(self.X, self.y):
            # Split data into training and validation sets
            X_train, y_train = self.X.iloc[train_index], self.y[train_index]
            X_val, y_val = self.X.iloc[val_index], self.y[val_index]

            # Train CatBoost model
            optuna_model.fit(X_train, y_train)
            score = self.scorer(optuna_model, X_val, y_val)
            print(score)
            scores.append(score)

        return score

    def best_model(self, best_param):
        best_model = catboost.CatBoostClassifier(**best_param)
        best_model.fit(self.X, self.y)
        return best_model
    
class ObjectiveXGB:
    def __init__(self, kfold, X, y, metric, random_state):
        self.kfold = kfold
        self.X = X
        self.y = y
        if metric in SCORERS:
            self.scorer = get_scorer(metric)
        elif metric in globals() and callable(globals()[metric]):
            self.scorer = make_scorer(globals()[metric], greater_is_better=False)
        else:
            # Here you can add more custom scorers if you need
            raise ValueError(f"Unsupported metric: {metric}")
        self.random_state = random_state

    def __call__(self, trial):        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'booster': 'gbtree',
            'tree_method': 'exact',
            'eta': trial.suggest_float('eta', 1e-5, 1, log=True),
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.1, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'alpha': trial.suggest_float('alpha', 0, 10),
            'lambda': trial.suggest_float('lambda', 0, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'nthread': -1,
            'seed': 42
        }
        optuna_model = xgb.XGBClassifier(**params) #, cv=self.kfold)
        scores = []

        for train_index, val_index in self.kfold.split(self.X,self.y):
            # Split data into training and validation sets
            X_train, y_train = self.X.iloc[train_index], self.y[train_index]
            X_val, y_val = self.X.iloc[val_index], self.y[val_index]
            

            # Train XGBoost model
            optuna_model.fit(X_train,y_train, verbose=0)
            score = self.scorer(optuna_model, X_val, y_val)
            print(score)
            scores.append(score)

        return score
    
    def best_model(self, best_param):
        best_model = xgb.XGBClassifier(**best_param)
        best_model.fit(self.X,self.y)
        return best_model
    
class ObjectiveLGBM:
    def __init__(self, kfold, X, y, metric, random_state):
        self.kfold = kfold
        self.X = X
        self.y = y
        if metric in SCORERS:
            self.scorer = get_scorer(metric)
        elif metric in globals() and callable(globals()[metric]):
            self.scorer = make_scorer(globals()[metric], greater_is_better=False)
        else:
            # Here you can add more custom scorers if you need
            raise ValueError(f"Unsupported metric: {metric}")
        self.random_state = random_state

    def __call__(self, trial):        
        params = {
            'objective': 'binary',
            'metric': 'logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
            'random_state': self.random_state,
            'verbose': -1,
        }
        optuna_model = lgb.LGBMClassifier(**params)
        scores = []

        for train_index, val_index in self.kfold.split(self.X,self.y):
            # Split data into training and validation sets
            X_train, y_train = self.X.iloc[train_index], self.y[train_index]
            X_val, y_val = self.X.iloc[val_index], self.y[val_index]
            

            # Train XGBoost model
            optuna_model.fit(X_train,y_train)
            score = self.scorer(optuna_model, X_val, y_val)
            print(score)
            scores.append(score)
        
        return score
    
    def best_model(self, best_param):
        best_model = lgb.LGBMClassifier(**best_param)
        best_model.fit(self.X,self.y)
        return best_model

#Regressor 
class ObjectiveRFRegressor:
    def __init__(self, kfold, X, y, metric, random_state):
        self.kfold = kfold
        self.X = X
        self.y = y
        if metric in SCORERS:
            self.scorer = get_scorer(metric)
        elif metric in globals() and callable(globals()[metric]):
            self.scorer = make_scorer(globals()[metric], greater_is_better=False)
        else:
            # Here you can add more custom scorers if you need
            raise ValueError(f"Unsupported metric: {metric}")
        self.random_state = random_state

    def __call__(self, trial):        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 100),
            'criterion': trial.suggest_categorical('criterion', ['squared_error']),
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': self.random_state
        }
        optuna_model = RandomForestRegressor(**params)
        scores = []

        for train_index, val_index in self.kfold.split(self.X, self.y):
            # Split data into training and validation sets
            X_train, y_train = self.X.iloc[train_index], self.y[train_index]
            X_val, y_val = self.X.iloc[val_index], self.y[val_index]

            # Train RandomForest model
            optuna_model.fit(X_train, y_train)
            score = self.scorer(optuna_model, X_val, y_val)
            print(score)
            scores.append(score)

        return score
    
    def best_model(self, best_param):
        best_model = RandomForestRegressor(**best_param)
        best_model.fit(self.X, self.y)
        return best_model

class ObjectiveCatBoostRegressor:
    def __init__(self, kfold, X, y, metric, random_state):
        self.kfold = kfold
        self.X = X
        self.y = y
        if metric in SCORERS:
            self.scorer = get_scorer(metric)
        elif metric in globals() and callable(globals()[metric]):
            self.scorer = make_scorer(globals()[metric], greater_is_better=False)
        else:
            # Here you can add more custom scorers if you need
            raise ValueError(f"Unsupported metric: {metric}")
        self.random_state = random_state

    def __call__(self, trial):
        params = {
            'task_type': 'CPU',
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'iterations': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1, log=True),
            'depth': trial.suggest_int('depth', 1, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 100, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'border_count': trial.suggest_int('border_count', 1, 255),
            'verbose': 0,
            'random_seed': 42
        }
        optuna_model = catboost.CatBoostRegressor(**params)
        scores = []

        for train_index, val_index in self.kfold.split(self.X, self.y):
            # Split data into training and validation sets
            X_train, y_train = self.X.iloc[train_index], self.y[train_index]
            X_val, y_val = self.X.iloc[val_index], self.y[val_index]

            # Train CatBoost model
            optuna_model.fit(X_train, y_train)
            score = self.scorer(optuna_model, X_val, y_val)
            print(score)
            scores.append(score)

        return score

    def best_model(self, best_param):
        best_model = catboost.CatBoostRegressor(**best_param)
        best_model.fit(self.X, self.y)
        return best_model

class ObjectiveXGBRegressor:
    def __init__(self, kfold, X, y, metric, random_state):
        self.kfold = kfold
        self.X = X
        self.y = y
        if metric in SCORERS:
            self.scorer = get_scorer(metric)
        elif metric in globals() and callable(globals()[metric]):
            self.scorer = make_scorer(globals()[metric], greater_is_better=False)
        else:
            # Here you can add more custom scorers if you need
            raise ValueError(f"Unsupported metric: {metric}")
        self.random_state = random_state

    def __call__(self, trial):        
        params = {
            'n_estimators' : trial.suggest_int('n_estimators', 80, 400),
            'objective': 'reg:squarederror',
            'eval_metric': 'logloss',
            'booster': 'gbtree',
            'tree_method': 'exact',
            'eta': trial.suggest_float('eta', 1e-5, 1, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'subsample': trial.suggest_float('subsample', 0.1, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'alpha': trial.suggest_float('alpha', 0, 10),
            'lambda': trial.suggest_float('lambda', 0, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'nthread': -1,
            'seed': 42
        }
        optuna_model = xgb.XGBRegressor(**params)
        scores = []

        for train_index, val_index in self.kfold.split(self.X, self.y):
            # Split data into training and validation sets
            X_train, y_train = self.X.iloc[train_index], self.y[train_index]
            X_val, y_val = self.X.iloc[val_index], self.y[val_index]

            # Train XGBoost model
            optuna_model.fit(X_train, y_train)
            score = self.scorer(optuna_model, X_val, y_val)
            #print(score)
            scores.append(score)

        return score

    def best_model(self, best_param):
        best_model = xgb.XGBRegressor(**best_param)
        best_model.fit(self.X, self.y)
        return best_model

class ObjectiveLGBMRegressor:
    def __init__(self, kfold, X, y, metric, random_state):
        self.kfold = kfold
        self.X = X
        self.y = y
        if metric in SCORERS:
            self.scorer = get_scorer(metric)
        elif metric in globals() and callable(globals()[metric]):
            self.scorer = make_scorer(globals()[metric], greater_is_better=False)
        else:
            # Here you can add more custom scorers if you need
            raise ValueError(f"Unsupported metric: {metric}")
        self.random_state = random_state

    def __call__(self, trial):        
        params = {
            'objective': 'regression',
            'metric': 'logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
            'random_state': self.random_state,
            'verbose': -1,
        }
        optuna_model = lgb.LGBMRegressor(**params)
        scores = []

        for train_index, val_index in self.kfold.split(self.X, self.y):
            # Split data into training and validation sets
            X_train, y_train = self.X.iloc[train_index], self.y[train_index]
            X_val, y_val = self.X.iloc[val_index], self.y[val_index]

            # Train LightGBM model
            optuna_model.fit(X_train, y_train)
            score = self.scorer(optuna_model, X_val, y_val)
            print(score)
            scores.append(score)

        return score
    
    def best_model(self, best_param):
        best_model = lgb.LGBMRegressor(**best_param)
        best_model.fit(self.X, self.y)
        return best_model

class ObjectiveLassoRegressor:
    def __init__(self, kfold, X_train, y_train,metric,random_state):
        self.kfold = kfold
        self.X_train = X_train
        self.y_train = y_train
        if metric == 'rmse':
            self.scorer = make_scorer(root_mean_squared_error, greater_is_better=False)
        else:
            self.scorer = get_scorer(metric)

    def __call__(self, trial):
        params = {
            'alpha': trial.suggest_loguniform('alpha', 1e-5, 1.0),
            'random_state': self.random_state
        }
        optuna_model = Lasso(**params)
        scores = []

        for train_index, val_index in self.kfold.split(self.X_train, self.y_train):
            X_train_cv, y_train_cv = self.X_train.iloc[train_index], self.y_train[train_index]
            X_val_cv, y_val_cv = self.X_train.iloc[val_index], self.y_train[val_index]

            optuna_model.fit(X_train_cv, y_train_cv)
            val_preds = optuna_model.predict(X_val_cv)
            score = self.scorer(y_val_cv, val_preds)
            scores.append(score)

        return score

    def best_model(self, best_param):
        best_model = Lasso(**best_param)
        best_model.fit(X_train_scaled, self.y_train)
        return best_model


class MyOptimizer:
    def __init__(self, objective, direction):
        self.objective = objective
        self.direction = direction
        self.study = optuna.create_study(direction=direction)

    def optimize(self, n_trials, callbacks=None):
        self.study.optimize(self.objective, n_trials=n_trials, callbacks=callbacks)
        
    def best_params(self):
        return self.study.best_params

    def best_value(self):
        return self.study.best_value


def ModelDictProblemType(cfg: DictConfig)-> None:
        
    if cfg.problem_type=='classification':
        models = {'random_forest': RandomForestClassifier(),
                  'XGBoost': xgb.XGBClassifier(),
                  'LGBM': lgb.LGBMClassifier()
        }
    elif cfg.problem_type=='regression':
        models = {'random_forest': RandomForestRegressor(),
                  'XGBoost': xgb.XGBRegressor(),
                  'LGBM': lgb.LGBMRegressor()
        }
    return models
