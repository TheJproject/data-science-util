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
    def __init__(self, apply_smote=True):
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
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.INIT_FEATS = ['Mean_Integrated', 'SD', 'EK', 'Skewness', 'Mean_DMSNR_Curve', 'SD_DMSNR_Curve', 'EK_DMSNR_Curve', 'Skewness_DMSNR_Curve']
        self.new_column_names = []
        self.first_call = True

    def get_unique_feature_names(self,input_features, feature_names):
        new_feature_names = []
        for i, feature_name in enumerate(feature_names):
            if feature_name in input_features:
                suffix = 1
                unique_name = f"{feature_name}_{suffix}"
                while unique_name in input_features:
                    suffix += 1
                    unique_name = f"{feature_name}_{suffix}"
                new_feature_names.append(unique_name)
            else:
                new_feature_names.append(feature_name)
        return new_feature_names
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        new_features = []

        if self.first_call:
            for feat1, feat2 in combinations(self.INIT_FEATS, 2):
                new_col_x = f"{feat1}#x#{feat2}"
                new_col_add = f"{feat1}#+#{feat2}"
                new_col_gt = f"{feat1}#>#{feat2}"

                new_features.append(pd.DataFrame({new_col_x: X[feat1] * X[feat2], new_col_add: X[feat1] + X[feat2], new_col_gt: X[feat1] > X[feat2]}))

                self.new_column_names.extend([new_col_x, new_col_add, new_col_gt])

            result = list(set(list(combinations(self.INIT_FEATS, 2)) + list(permutations(self.INIT_FEATS, 2))))

            for feat1, feat2 in result:
                new_col_div = f"{feat1}#/#{feat2}"
                new_features.append(pd.DataFrame({new_col_div: X[feat1] / X[feat2]}))
                new_features[-1].replace(np.inf, 1e15, inplace=True)

                self.new_column_names.append(new_col_div)

            self.new_column_names = self.get_unique_feature_names(self.INIT_FEATS, self.new_column_names)
            self.first_call = False
        else:
            for col in self.new_column_names:
                col_split = col.split('#')
                op = col_split[1]
                feat1 = col_split[0]
                feat2 = col_split[2]

                if op == 'x':
                    new_features.append(pd.DataFrame({col: X[feat1] * X[feat2]}))
                elif op == '+':
                    new_features.append(pd.DataFrame({col: X[feat1] + X[feat2]}))
                elif op == '>':
                    new_features.append(pd.DataFrame({col: X[feat1] > X[feat2]}))
                elif op == '/':
                    new_features.append(pd.DataFrame({col: X[feat1] / X[feat2]}))
                    new_features[-1].replace(np.inf, 1e15, inplace=True)

        X = pd.concat([X] + new_features, axis=1)
        return X

    def get_new_feature_names(self):
        return self.INIT_FEATS + self.new_column_names


        
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