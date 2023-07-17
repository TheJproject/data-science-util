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
# 
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim_small, encoding_dim_large):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim_large),
            nn.ReLU(True),
            nn.Linear(encoding_dim_large, encoding_dim_small),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim_small, encoding_dim_large),
            nn.ReLU(True),
            nn.Linear(encoding_dim_large, input_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class ObjectiveAE(object):
    def __init__(self, kfold, X, y, metric, random_state):
        self.kfold = kfold
        self.random_state = random_state
        self.metric = metric
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X = X
        self.y = y
        self.epochs = 1000  # Modify this to your requirement
        self.batch_size = 128  # Modify this based on your hardware
        self.best_model = None
        self.min_loss = np.inf

    def create_model(self, trial):
        input_size = self.X.shape[1]
        hidden_size_small = trial.suggest_int('hidden_size_small', 3, 16)
        hidden_size_large = trial.suggest_int('hidden_size_large', 32, 200)

        model = Autoencoder(input_size, hidden_size_small,hidden_size_large)
        return model

    def create_optimizer(self, trial, model):
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        return optimizer

    def train(self):
        self.model.train()
        running_loss = 0
        for inputs, in self.train_loader:  # Notice the comma here to unpack the list
            self.optimizer.zero_grad()
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, inputs)  # Notice here that we use the inputs as targets
            running_loss += loss.item() * inputs.size(0)
            loss.backward()
            self.optimizer.step()
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def validate(self):
        self.model.eval()
        running_loss = 0
        for inputs, in self.valid_loader:  # Notice the comma here to unpack the list
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)  # Also here, inputs are used as targets
                running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(self.valid_loader.dataset)
        if epoch_loss < self.min_loss:
            self.min_loss = epoch_loss
            self.best_model = copy.deepcopy(self.model)
        return epoch_loss


    def __call__(self, trial):
        self.model = self.create_model(trial).to(self.device)
        self.optimizer = self.create_optimizer(trial, self.model)
        self.criterion = nn.MSELoss()  # for autoencoder

        kf = self.kfold
        
        # Since we don't have labels for AE, we can just fill an array with ones for the sake of splitting
        dummy_y = torch.ones(len(self.X))

        for fold, (train_index, val_index) in enumerate(kf.split(self.X, dummy_y)):
            print(f"Fold {fold+1}")

            X_train_fold = torch.tensor(self.X.values[train_index], dtype=torch.float32)
            X_val_fold = torch.tensor(self.X.values[val_index], dtype=torch.float32)

            train_dataset = TensorDataset(X_train_fold)
            val_dataset = TensorDataset(X_val_fold)

            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            self.valid_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

            # Early stopping
            early_stopping = EarlyStopping(patience=3, verbose=True)

            for epoch in range(self.epochs):
                train_loss = self.train()
                valid_loss = self.validate()

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

        return -self.min_loss


# Once you've found the best parameters for the autoencoder, you can create your final model:

class EncoderClassifier(nn.Module):
    def __init__(self, autoencoder, hidden_size, output_size):
        super(EncoderClassifier, self).__init__()
        self.encoder = autoencoder.encoder
        for param in self.encoder.parameters():
            param.requires_grad = False  # freeze the encoder
        self.classifier = Classifier(hidden_size, hidden_size, output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


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

        score = val_loss

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
                inputs = inputs#.to(device)
                labels = labels#.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return loss.item()
