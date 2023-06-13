import pyarrow.feather as feather
import pandas as pd
import numpy as np
import hydra

#from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder,PolynomialFeatures

from itertools import combinations, permutations
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

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

class AutoFeatureTransform(BaseEstimator, TransformerMixin):
    def __init__(self, problem_type, **kwargs):
        self.problem_type = problem_type
        self.n_features = 5
        self.indices = None
        self.poly = PolynomialFeatures(2, interaction_only=True, include_bias=False)

    def fit(self, X, y=None):
        new_data = self._generate_features(X, fit=True)
        if self.problem_type == 'regression':
            model = RandomForestRegressor()
        elif self.problem_type == 'classification':
            model = RandomForestClassifier()
        else:
            raise ValueError('Problem_type should be either regression or classification. Check the value in the config file')
        
        model.fit(new_data, y)
        print("Finding the best features...")
        result = permutation_importance(model, new_data, y, n_repeats=2, random_state=0)
        
        self.indices = result.importances_mean.argsort()[::-1][:self.n_features]
        self.column_names_ = new_data.columns[self.indices].tolist()
        
        return self

    
    def _generate_features(self, X, fit=False):
        print("Generating new features...")
        if fit:
            X_poly = self.poly.fit_transform(X)
        else:
            X_poly = self.poly.transform(X)
        
        # Get the feature names from PolynomialFeatures
        all_feature_names = self.poly.get_feature_names_out(X.columns)
        
        # The original features are the ones that are not combined with others.
        # These features are represented as single terms like 'x0', 'x1', etc. in the output of get_feature_names_out.
        # We want to exclude these original features from the new_data DataFrame.
        interaction_feature_names = [name for name in all_feature_names if ' ' in name]
        
        # Get the indices of the interaction features in the transformed data
        interaction_feature_indices = [i for i, name in enumerate(all_feature_names) if name in interaction_feature_names]
        
        # Create a DataFrame for the transformed data
        new_data = pd.DataFrame(X_poly[:, interaction_feature_indices], columns=interaction_feature_names)
        
        print("Type X-poly :" + str(type(X_poly)))
        
        return new_data

    def transform(self, X):
        new_data = self._generate_features(X, fit=False)
        selected_columns = new_data.columns[self.indices]
        return new_data[selected_columns]
    
    def get_transformed_columns(self):
        return self.column_names_

class ExternalFeatureTemplate(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        pass 

    def fit(self, X, y=None):
        new_data = self._generate_features(X)
        return self

    def _generate_features(self, X, fit=False):
        print("Generating new features...")
        new_data = X.copy()  # Assuming we make some transformations on X
        return new_data

    def transform(self, X):
        new_data = self._generate_features(X, fit=False)
        selected_columns = new_data.columns[self.indices]
        return new_data[selected_columns]

    def get_transformed_columns(self):
        return self.column_names_
    
class CustomFeatureTransform(BaseEstimator, TransformerMixin):
    def __init__(self, X,apply_smote=True):
        self.apply_smote = apply_smote
        self.smote = SMOTE(random_state=42)

    def fit(self, X, y=None):
        if self.apply_smote and y is not None:
            self.smote.fit_resample(X, y)
        
        return self
    
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
        

def no_feature(data):
    # transformation code
    pass


def custom_feature(df):
    """
    pressure: Pressure
    mass_flux: Mass flux
    D_e: Inner diameter
    D_h: Horizontal diameter
    length: Length
    chf_exp: Critical heat flux in experiments
    """
    # Mass flux and length product
    # df['mass_length_product'] = df['mass_flux'] * df['length']
    
    # Adiabatic surface area
    df['adiabatic_surface_area'] = df['D_e'] * df['length']
    
    # Surface area to horizontal diameter ratio
    df['surface_diameter_ratio'] = df['D_e'] / df['D_h']
    
    # Pressure to mass flux ratio
    # df['pressure_flux_ratio'] = df['pressure'] / df['mass_flux']
    
    # Combinations of features: create products or ratios between features
    # df['length_surface_ratio'] = df['length'] / df['adiabatic_surface_area']
    
    return df

def auto_feature(data):
    # transformation code
    pass

def ext_feature(data):
    # transformation code
    pass

FEATURES_MAPPING = {
    "no_feature": no_feature,
    "custom_feature": custom_feature,
    "auto_feature": auto_feature,
    "ext_feature": ext_feature,
}