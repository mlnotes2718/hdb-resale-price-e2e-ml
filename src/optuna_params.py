# config.py
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, Ridge

## Optuna Grid Search CV
params_config = {
    "XGBoost_tuned": {
        "estimator": xgb.XGBRegressor(n_jobs=-1),
        "param_distributions": {
            "regressor__subsample":         FloatDistribution(low=0.72, high=0.88),
            "regressor__reg_lambda":        IntDistribution(low=1,    high=1,    step=1),
            "regressor__reg_alpha":         IntDistribution(low=1,    high=2,    step=1),
            "regressor__n_estimators":      IntDistribution(low=160,  high=240,  step=1),
            "regressor__max_depth":         IntDistribution(low=8,    high=10,   step=1),
            "regressor__learning_rate":     FloatDistribution(low=0.09, high=0.11),
            "regressor__gamma":             IntDistribution(low=1,    high=1,    step=1),
            "regressor__colsample_bytree":  FloatDistribution(low=0.72, high=0.88),
        },
        "cv": 5,
        "trials": 60,
        "scoring": "neg_mean_absolute_error",
    },

    "LightGBM_tuned": {
            "estimator": lgb.LGBMRegressor(verbose=-1, n_jobs=-1),
            "param_distributions": {
                "regressor__subsample":         FloatDistribution(low=0.9,  high=1.0, log=False),
                "regressor__reg_lambda":        IntDistribution(low=1,    high=2,   step=1),
                "regressor__reg_alpha":         IntDistribution(low=1,    high=1,   step=1),
                "regressor__num_leaves":        IntDistribution(low=102,  high=152, step=1),
                "regressor__n_estimators":      IntDistribution(low=80,   high=120, step=1),
                "regressor__min_child_samples": IntDistribution(low=16,   high=24,  step=1),
                "regressor__max_depth":         IntDistribution(low=8,    high=10,  step=1),
                "regressor__learning_rate":     FloatDistribution(low=0.18, high=0.22, log=False),
                "regressor__colsample_bytree":  FloatDistribution(low=0.9,  high=1.0, log=False),
            },
            "cv": 5,
            "trials": 30,
            "scoring": "neg_mean_absolute_error",
        },
}

# Optuna Grid Search CV for Linear Regression
params_config_linear = {
    "Linear_Regression_tuned": {
        "estimator": LinearRegression(),
        "param_distributions": {
            "preprocessor__num__poly__degree": IntDistribution(low=1, high=3),
            "regressor__fit_intercept": CategoricalDistribution([True, False]),
        },
        "cv": 5,
        "trials": 10,
        "scoring": "neg_mean_absolute_error",
    },
    "Ridge_Regression_tuned": {
        "estimator": Ridge(),
        "param_distributions": {
            "preprocessor__num__poly__degree": IntDistribution(low=1, high=3),
            "regressor__fit_intercept": CategoricalDistribution([True, False]),
            "regressor__alpha": FloatDistribution(low=0.001, high=100.0, log=True),
            "regressor__solver": CategoricalDistribution(["auto", "svd", "cholesky", "lsqr", "sparse_cg"]),
        },
        "cv": 5,
        "trials": 10,
        "scoring": "neg_mean_absolute_error",
    },
}