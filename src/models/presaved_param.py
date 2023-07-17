# presaved_param.py

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb

param_rf_cla = {'n_estimators': 100, 'max_depth': 5}
param_catboost_cla = {'depth': 7, 'learning_rate': 0.1}
param_xgb_cla = {'eta': 0.09997487669624248, 'max_depth': 9, 'subsample': 0.4503180982597529, 'colsample_bytree': 0.28852184628909433, 'gamma': 0.048619898711735665, 'alpha': 1.037873925072671, 'lambda': 
2.2991082165191634, 'min_child_weight': 18} #{'n_estimators': 100, 'max_depth': 5}
param_lgbm_cla = {'n_estimators': 100, 'max_depth': 5}

param_rf_reg = {'n_estimators': 100, 'max_depth': 5}
param_catboost_reg = {'depth': 7, 'learning_rate': 0.1}
param_xgb_reg = {'n_estimators': 3000,
        'max_depth': 4,
        'learning_rate': 0.4,
        'subsample': 0.8,
        'colsample_bytree': 1,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'rmse',
        'verbosity': 0,
        'early_stopping_rounds': 100
        }
param_lgbm_reg = {'n_estimators': 100, 'max_depth': 5}

model_dict = {
    "classification": {
        "RF": RandomForestClassifier(**param_rf_cla),
        "CatBoost": CatBoostClassifier(**param_catboost_cla),
        "XGBoost": xgb.XGBClassifier(**param_xgb_cla),
        "LGBM": lgb.LGBMClassifier(**param_lgbm_cla),
    },
    "regression": {
        "RF": RandomForestRegressor(**param_rf_reg),
        "CatBoost": CatBoostRegressor(**param_catboost_reg),
        "XGBoost": xgb.XGBRegressor(**param_xgb_reg),
        "LGBM": lgb.LGBMRegressor(**param_lgbm_reg),
    },
}
