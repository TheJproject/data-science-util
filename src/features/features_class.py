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

class AutoFeatureTransform(BaseEstimator, TransformerMixin):
    def __init__(self, problem_type, **kwargs):
        self.problem_type = problem_type
        self.n_features = 1
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
        data_with_new_features = pd.concat([X, new_data], axis=1)
        print("Type X-poly :" + str(type(data_with_new_features)))
        
        return new_data

    def transform(self, X):
        new_data = self._generate_features(X, fit=False)
        selected_columns = new_data.columns[self.indices]
        #new_data = new_data.reindex(X.index)
        # print head of new data and X
        print("Head of new data: " + str(new_data.head()))
        print("Head of X: " + str(X.head()))



        print("Shape of new data: " + str(new_data.shape))
        print("Shape of X: " + str(X.shape))
        X.reset_index(drop=True, inplace=True)
        new_data.reset_index(drop=True, inplace=True)
        return pd.concat([X, new_data[selected_columns]], axis=1)
        return new_data[selected_columns]
        #return new_data[selected_columns]
    
    def get_transformed_columns(self):
        return self.column_names_
    
    def set_output(self, transform="passthrough"):
        self.output_transform = transform
        return self

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
    

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class PCATransformer(BaseEstimator, TransformerMixin):
    def __init__(self, problem_type, n_components=2, **kwargs):
        self.problem_type = problem_type
        self.n_components = n_components
        self.indices = None
        self.pca = PCA(n_components=self.n_components)

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

        self.indices = result.importances_mean.argsort()[::-1][:self.n_components]
        self.column_names_ = new_data.columns[self.indices].tolist()

        return self

    def _generate_features(self, X, fit=False):
        print("Applying PCA...")
        if fit:
            self.pca.fit(X)
        new_data_pca = self.pca.transform(X)

        new_data = pd.DataFrame(new_data_pca, columns=[f"PC_{i+1}" for i in range(self.n_components)])
        return new_data

    def transform(self, X):
        new_data = self._generate_features(X, fit=False)
        selected_columns = new_data.columns[self.indices]
        return new_data[selected_columns]

    def get_transformed_columns(self):
        return self.column_names_

        
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
        

def no_feature(df):
    return df


def custom_feature(df):
    """
    pressure: Pressure
    mass_flux: Mass flux
    D_e: Inner diameter
    D_h: Horizontal diameter
    length: Length
    chf_exp: Critical heat flux in experiments
    """

    # transformation code

    #df['SD x EK'] = df['SD'] * df['EK']
    # Mass flux and length product
    # df['mass_length_product'] = df['mass_flux'] * df['length']
    
    # Adiabatic surface area
    #df['adiabatic_surface_area'] = df['D_e'] * df['length']
    
    # Surface area to horizontal diameter ratio
    #df['surface_diameter_ratio'] = df['D_e'] / df['D_h']
    
    # Pressure to mass flux ratio
    # df['pressure_flux_ratio'] = df['pressure'] / df['mass_flux']
    
    # Combinations of features: create products or ratios between features
    # df['length_surface_ratio'] = df['length'] / df['adiabatic_surface_area']
    
    return df

def ext_feature(data):
    # transformation code
    pass

FEATURES_MAPPING = {
    "no_feature": no_feature,
    "custom_feature": custom_feature,
    "ext_feature": ext_feature,
}   