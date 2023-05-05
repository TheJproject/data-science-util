import pyarrow.feather as feather
import pandas as pd
import numpy as np
import hydra

from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from itertools import combinations, permutations
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

class FeatureNoTransform():
    def __init__(self, data_path):
        self.data = feather.read_feather(data_path)
    def transform(self):
        df = self.data
        pass

class AutoFeatureTransform():
    def __init__(self, data):
        self.data = data

    def transform(self, method):
        pass

class CustomFeatureTransform(BaseEstimator, TransformerMixin):
    def __init__(self, X,apply_smote=True):
        self.apply_smote = apply_smote
        self.smote = SMOTE(random_state=42)

    def fit(self, X, y=None):
        #if self.apply_smote and y is not None:
        #    self.smote.fit(X, y)
        
        return self

    def transform(self, X, y=None):
        # Perform custom feature engineering here
        # Example: X['new_feature'] = X['feature1'] * X['feature2']

        #if self.apply_smote and y is not None:
        #    X_resampled, _ = self.smote.transform(X, y)
        #    return X_resampled
        #else:
            return X
    
class CustomFeatureTransformFS(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.FEATS = []

    def fit(self, X, y=None):
        feat = X.columns.tolist()
        self.FEATS.extend(feat)
        return self

    def transform(self, X, y=None):
        print(X)
        feat = X.columns.tolist()
        print("len of feat inside trasnform :" + str(len(feat)))
        X = X.copy()
        X['Mean_Integrated_x_SD'] = X['Mean_Integrated'] * X['SD']
        #for feat1, feat2 in combinations(self.INIT_FEATS, 2):
        #    X[f'{feat1}_x_{feat2}'] = X[f'{feat1}'] * X[f'{feat2}']
        #    X[f'{feat1}_+_{feat2}'] = X[f'{feat1}'] + X[f'{feat2}']
        #    X[f'{feat1}_>_{feat2}'] = X[f'{feat1}'] > X[f'{feat2}']
        print("shape of X inside transform:" + str(X.shape))
        for new_feat in ['Mean_Integrated_x_SD']:
            if new_feat not in self.FEATS:
                self.FEATS.extend([new_feat])
        return X

    def get_feature_names_out(self):
        return self.FEATS

        
class ExternalFeatureTransform():
    def __init__(self, data):
        self.data = data

    def transform(self, method):
        pass

class CustomTargetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = LabelEncoder()
        self.is_classification = None

    def fit(self, y, X=None):
        if y.dtype == np.number:
            self.is_classification = False
        else:
            self.is_classification = True
            self.encoder.fit(y)
        return self

    def transform(self, y, X=None):
        if self.is_classification:
            return self.encoder.transform(y)
        else:
            return y

    def inverse_transform(self, y, X=None):
        if self.is_classification:
            return self.encoder.inverse_transform(y)
        else:
            return y