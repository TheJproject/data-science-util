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
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn.functional as F

from omegaconf import DictConfig
# Assuming that your_script.py and features_class.py are siblings in the directory structure
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../features'))
# Now the script should be able to import features_class
from features_class import AutoFeatureTransform
from sklearn.base import BaseEstimator, TransformerMixin

# The rest of your script...

log = logging.getLogger(__name__)
#wandb.init()
class ShapePrinter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.output_transform = None

    def fit(self, X, y=None):
        # Remember the feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns
        else:
            self.feature_names_ = [f'col_{i}' for i in range(X.shape[1])]
        return self

    def transform(self, X):
        print(f"ShapePrinter: {X.shape}")
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_

    def set_output(self, transform):
        self.output_transform = transform
        return self


class EnsembleModel(BaseEstimator, ClassifierMixin):
    def __init__(self, models, is_pytorch):
        self.models = models
        self.is_pytorch = is_pytorch

    def predict_proba(self, X):
        proba_predictions = []

        for i, model in enumerate(self.models):
            if self.is_pytorch[i]:
                # For PyTorch models
                X_tensor = torch.tensor(X.values, dtype=torch.float32)
                output = model(X_tensor)
                prediction = F.softmax(output, dim=1).detach().numpy()
            else:
                # For scikit-learn models
                prediction = model.predict_proba(X)
            proba_predictions.append(prediction)
        
        return np.mean(proba_predictions, axis=0)

    def predict(self, X):
        # The predicted class will be the one with highest average probability
        return np.argmax(self.predict_proba(X), axis=1)

    
@hydra.main(config_path="../..",config_name="config")
def main(cfg: DictConfig) -> None:
    working_dir = os.getcwd()
    data_dir = hydra.utils.to_absolute_path(cfg.data_dir)
    data_path = os.path.join(data_dir, f'cleaned/{cfg.competition_name}/test.feather')
    model_dir = hydra.utils.to_absolute_path(cfg.model_dir)
    print(f"The current working directory for data is {data_dir}")
    print(f"The current working directory is {working_dir}")
    print(f"The current working directory for models is {model_dir}")
    log.info(f"Experiment: {working_dir}")

    if len(cfg.ensemble_list) > 1:
        print("Chose ensemble model")
        model_path = os.path.join(model_dir, f'{cfg.competition_name}/models_{cfg.selected_model}/stacking_model.pkl')
        model = joblib.load(model_path)
    else:
        print("Chose single model")
        model_path = os.path.join(model_dir, f'{cfg.competition_name}/models_{cfg.selected_model}/{cfg.ensemble_list[0]}_model.pkl')
        model = joblib.load(model_path)

    #Load the feature_engineering pipeline for the test set 
    preprocessing_path = os.path.join(model_dir, f'{cfg.competition_name}/pipeline.joblib')
    feature_engineering = joblib.load(preprocessing_path)

    df = feather.read_feather(data_path)

    df.columns = df.columns.str.replace('\[.*?\]', '', regex=True)
    df.columns = df.columns.str.strip()
    
    print(df.head())
    print(df.info())
    print("\nBefore feature engineering, data shape: ", df.shape)

    df_engineered= feature_engineering.transform(df)
    output_feature_names = feature_engineering.get_feature_names_out()
    df_engineered = pd.DataFrame(df_engineered, columns=output_feature_names)
    
    print(df_engineered.info())
  
    print("\nAfter feature engineering, data shape: ", df_engineered.shape)
    # Make prediction on test data
    predictions = model.predict(df_engineered)
    # Make prediction on test data
    predictions_proba = model.predict_proba(df_engineered)[:, 1]  # Get the probability of the second class
    print(predictions_proba)
    print(predictions)#[:,1])
    # Create submission file
    submission_df = pd.DataFrame({'id': df.index, cfg.target_col: predictions_proba})#[:,1]})
    submission_path = os.path.join(data_dir, f'submission/{cfg.competition_name}')
    os.makedirs(submission_path , exist_ok=True)
    submission_df.to_csv(submission_path + '/submission.csv', index=False)


if __name__ == "__main__":
    main()
