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
import copy

from omegaconf import DictConfig

# objectives.py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import optuna
import numpy as np
#Additional Scorer

def rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred,squared=False)
    return -mse

import numpy as np

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if not actual:
        return 0.0

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

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

        return np.mean(scores)
        
    def best_model(self, best_param):
        best_model = RandomForestClassifier(**best_param)
        best_model.fit(self.X,self.y)
        return best_model

class ObjectiveCatBoost:
    def __init__(self, kfold, X, y, metric, random_state):
        self.kfold = kfold
        self.X = X
        self.y = y
        self.metric = metric
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
            'eval_metric': 'AUC',
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

        return np.mean(scores)

    def best_model(self, best_param):
        best_model = catboost.CatBoostClassifier(**best_param)
        best_model.fit(self.X, self.y)
        return best_model
    
class ObjectiveXGB:
    def __init__(self, kfold, X, y, metric, random_state):
        self.kfold = kfold
        self.X = X
        self.y = y
        self.metric = metric
        print(metric)
        if metric in SCORERS:
            self.scorer = get_scorer(metric)
        elif metric in globals() and callable(globals()[metric]):
            self.scorer = make_scorer(globals()[metric], greater_is_better=True)
        else:
            # Here you can add more custom scorers if you need
            raise ValueError(f"Unsupported metric: {metric}")
        self.random_state = random_state

    def __call__(self, trial):        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'booster': 'gbtree',
            'tree_method': 'exact',
            'eta': trial.suggest_float('eta', 1e-5, 1, log=True),
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.1, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
            'gamma': trial.suggest_float('gamma', 0, 2),
            'alpha': trial.suggest_float('alpha', 0, 10),
            'lambda': trial.suggest_float('lambda', 0, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'n_jobs': -1,
            'seed': self.random_state
        }
        optuna_model = xgb.XGBClassifier(**params) #, cv=self.kfold)
        scores = []

        for train_index, val_index in self.kfold.split(self.X,self.y):
            # Split data into training and validation sets
            X_train, y_train = self.X.iloc[train_index], self.y[train_index]
            X_val, y_val = self.X.iloc[val_index], self.y[val_index]
            

            # Train XGBoost model
            optuna_model.fit(X_train,y_train, verbose=1)
            score = self.scorer(optuna_model, X_val, y_val)
            print(score)
            scores.append(score)

        return np.mean(scores)
    
    def best_model(self, best_param):
        best_model = xgb.XGBClassifier(**best_param)
        best_model.fit(self.X,self.y)
        return best_model
    
class ObjectiveLGBM:
    def __init__(self, kfold, X, y, metric, random_state):
        self.kfold = kfold
        self.X = X
        self.y = y
        self.metric = metric
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
            'metric': 'auc',
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
        
        return np.mean(scores)
    
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

        return np.mean(scores)
    
    def best_model(self, best_param):
        best_model = RandomForestRegressor(**best_param)
        best_model.fit(self.X, self.y)
        return best_model

class ObjectiveCatBoostRegressor:
    def __init__(self, kfold, X, y, metric, random_state):
        self.kfold = kfold
        self.X = X
        self.y = y
        self.metric = metric
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
            'eval_metric': self.metric,
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

        return np.mean(scores)

    def best_model(self, best_param):
        best_model = catboost.CatBoostRegressor(**best_param)
        best_model.fit(self.X, self.y)
        return best_model

class ObjectiveXGBRegressor:
    def __init__(self, kfold, X, y, metric, random_state):
        self.kfold = kfold
        self.X = X
        self.y = y
        self.metric = metric
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
            'eval_metric': self.metric,
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

        return np.mean(scores)

    def best_model(self, best_param):
        best_model = xgb.XGBRegressor(**best_param)
        best_model.fit(self.X, self.y)
        return best_model

class ObjectiveLGBMRegressor:
    def __init__(self, kfold, X, y, metric, random_state):
        self.kfold = kfold
        self.X = X
        self.y = y
        self.metric = metric
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
            'metric': self.metric,
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

        return np.mean(scores)
    
    def best_model(self, best_param):
        best_model = lgb.LGBMRegressor(**best_param)
        best_model.fit(self.X, self.y)
        return best_model


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
        
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')  # <-- Here, modify & specify the path where you want to save the model.
        self.val_loss_min = val_loss


class ObjectiveNN(object):
    def __init__(self, kfold, X, y, metric, random_state):
        self.kfold = kfold
        self.metric = metric
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X = X
        self.y = y
        self.epochs = 10  # Modify this to your requirement
        self.batch_size = 128  # Modify this based on your hardware
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.best_model = None
        self.min_loss = np.inf

    def create_model(self, trial):
        input_size = self.X.shape[1]
        output_size = len(set(self.y))
        hidden_size = trial.suggest_int('hidden_size', 50, 200)
        model = Net(input_size, hidden_size, output_size)
        return model

    def create_optimizer(self, trial, model):
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        return optimizer

    def train(self, trial):
        self.model.train()
        running_loss = 0
        for inputs, targets in self.train_loader:
            self.optimizer.zero_grad()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            loss.backward()
            self.optimizer.step()
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def validate(self, trial):
        self.model.eval()
        running_loss = 0
        for inputs, targets in self.valid_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(self.valid_loader.dataset)
        if epoch_loss < self.min_loss:
            self.min_loss = epoch_loss
            self.best_model = copy.deepcopy(self.model)
        return epoch_loss


    def __call__(self, trial):
        self.model = self.create_model(trial).to(self.device)
        self.optimizer = self.create_optimizer(trial, self.model)
        self.criterion = nn.CrossEntropyLoss()  # for classification, use MSE loss for regression

        # Initialize the early_stopping object
        early_stopping = EarlyStopping(patience=3, verbose=True)

        kf = self.kfold

        for train_index, val_index in kf.split(self.X):
            X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]

            train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), 
                                        torch.tensor(y_train.values, dtype=torch.long)) 
            valid_dataset = TensorDataset(torch.tensor(X_val.values, dtype=torch.float32), 
                                        torch.tensor(y_val.values, dtype=torch.long))

            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

            for epoch in range(self.epochs):
                train_loss = self.train(trial)
                valid_loss = self.validate(trial)

                # Print progress
                print(f"Epoch {epoch+1}/{self.epochs}")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Validation Loss: {valid_loss:.4f}")

                # Early_stopping needs the validation loss to check if it has decresed, 
                # and if it has, it will make a checkpoint of the current model
                early_stopping(valid_loss, self.model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        return -valid_loss



class ObjectiveNNRegressor:
    def __init__(self, trial):
        self.trial = trial

    def __call__(self, X_train, y_train, device, n_epochs=100, batch_size=64):
        input_size = X_train.shape[1]
        output_size = 1

        # Defining the hyperparameters
        hidden_size = self.trial.suggest_int('hidden_size', 50, 200)
        lr = self.trial.suggest_loguniform('lr', 1e-5, 1e-1)
        
        model = Net(input_size, hidden_size, output_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()  

        train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), 
                                      torch.tensor(y_train.values, dtype=torch.float32)) 
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(n_epochs):
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return loss.item()



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



