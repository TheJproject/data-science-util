import pyarrow.feather as feather
import pandas as pd
import numpy as np
import hydra
import logging 

#from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder,PolynomialFeatures

from itertools import combinations, permutations
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer

from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import logging

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import logging

class SimpleOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, all_data=None, **kwargs):
        self.log = logging.getLogger(__name__)
        self.all_data = all_data

    def fit(self, X, y=None):
        # Check if X is a DataFrame, if not convert it to one
        if not isinstance(self.all_data, pd.DataFrame):
            self.all_data = pd.DataFrame(self.all_data)

        self.all_data = self.all_data.fillna('nan')
        
        # Identify the categorical columns
        self.categorical_columns = self.all_data.select_dtypes(include=['object']).columns.tolist()
        print("Categorical columns >IN FIT: " + str(self.categorical_columns))
        self.encoded_columns = pd.get_dummies(self.all_data, columns=self.categorical_columns).columns.tolist()
        return self

    def _generate_features(self, X):
        print("Generating new features...")
        X = X.fillna('nan')
    # Identify the categorical columns
        self.categorical_columns = self.all_data.select_dtypes(include=['object']).columns.tolist()
        print("Identified categorical columns during fit: ", self.categorical_columns)
    
        # Only encode columns present in both self.categorical_columns and X
        columns_to_encode = list(set(self.categorical_columns) & set(X.columns))
        X_encoded = pd.get_dummies(X, columns=columns_to_encode,dtype=float)
        print("Categorical columns >IN GENERATE FEATURES: " + str(X_encoded))




        #X_encoded = X_encoded[self.encoded_columns]
        print("ENCODED COLUMNS: " + str(self.encoded_columns))
        print("Type X-encoded :" + str(type(X_encoded)))
        return X_encoded



    def transform(self, X):
        # Convert X to a DataFrame if it isn't one already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        new_data = self._generate_features(X)
        print("Head of new data: " + str(new_data.head()))
        print("Head of X: " + str(X.head()))
        print("Shape of new data: " + str(new_data.shape))
        print("Shape of X: " + str(X.shape))
        print("Categorical columns >IN TRANSFORM: " + str(new_data.columns.tolist()))
        
        return new_data
    

    def get_transformed_columns(self):
        return self.encoded_columns

    def set_output(self, transform="passthrough"):
        self.output_transform = transform
        return self

    def get_feature_names_out(self, input_features=None):
        return self.encoded_columns





class AutoFeatureTransform(BaseEstimator, TransformerMixin):
    def __init__(self, model, problem_type, new_feature_count):
        self.model = model
        self.problem_type = problem_type
        self.new_feature_count = new_feature_count
        self.indices = None
        self.poly = PolynomialFeatures(2, interaction_only=False, include_bias=False)
        self.input_feature_names_ = None
        self.imputer = SimpleImputer(strategy='mean') # add imputer here

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        print(X.columns.tolist())
        # add imputation for missing values
        X_imputed = self.imputer.fit_transform(X)   

        # Store original feature names
        self.input_feature_names_ = X.columns.tolist()

        new_features = self._generate_features(X_imputed, fit=True)
        
        # Fit the model
        self.model.fit(new_features, y)

        # Turn off feature_names_in_
        self.model.feature_names_in_ = None

        # Conduct permutation importance
        result = permutation_importance(self.model, new_features, y, n_repeats=2, random_state=0)

        # make sure the new_feature_count doesn't exceed the number of generated features
        self.new_feature_count = min(self.new_feature_count, len(result.importances_mean))

        self.indices = result.importances_mean.argsort()[::-1][:self.new_feature_count]
        self.column_names_ = new_features.columns[self.indices].tolist()
        print("Column names: " + str(self.column_names_))
        
        return self



    def _generate_features(self, X, fit=False):
        if fit:
            X_poly = self.poly.fit_transform(X)
            all_feature_names = self.poly.get_feature_names_out(input_features=self.input_feature_names_)
            new_data = pd.DataFrame(data=X_poly, columns=all_feature_names)
        else:
            X_poly = self.poly.transform(X)
            new_data = pd.DataFrame(data=X_poly, columns=self.poly.get_feature_names_out(input_features=self.input_feature_names_))

        return new_data

    def transform(self, X):
        # If X is a numpy array, convert it to a DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.input_feature_names_)
        X = X.copy()
        # Impute missing values in X
        X = pd.DataFrame(self.imputer.transform(X), columns=self.input_feature_names_)

        new_data = self._generate_features(X, fit=False)

        if len(self.indices) > len(new_data.columns):
            raise ValueError("Indices length is greater than the number of new columns")

        # Use original column names and selected new ones
        selected_new_features = new_data[self.column_names_]
        transformed_data = pd.concat([X, selected_new_features], axis=1)
        print("column names end of transform: " + str(transformed_data.columns.tolist()))
        return transformed_data


    def get_feature_names_out(self, input_features=None):
        if self.column_names_ is not None:
            return self.input_feature_names_ + self.column_names_
        else:
            raise AttributeError("Transformer has not been fitted yet.")




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