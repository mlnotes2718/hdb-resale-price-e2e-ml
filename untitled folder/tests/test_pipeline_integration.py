import pandas as pd
import pytest
from data_preparation import DataPreparation
from model_training import ModelTraining
from optuna_params import params_config

@pytest.fixture
def full_config():
    return {
        **{"target_column": "price", "val_test_size": 0.5, "val_size": 0.5},
        **{"numerical_features": ["remaining_lease_by_months"], "nominal_features": ["flat_type", "town_name", 'flatm_name'], "passthrough_features": ["storey_range"]},
    }

@pytest.fixture
def raw_df():
    # build a small but complete raw HDB-like DataFrame
    return pd.DataFrame({
        "lease_commence_date": [10, -20, 30, 40],
        "flat_type": ["5 ROOM", "4 ROOM", "5 ROOM", "FOUR ROOM"],
        "town_id": [1, 1, 2, 2],
        "town_name": ["TOWN A", None, "TOWN B", None],
        "flatm_id": [100, 100, 200, 200],
        "flatm_name": ["TYPE X", None, "TYPE Y", None],
        "storey_range": ["01 TO 03"]*4,
        "month": ["2020-01", "2020-02", "2020-03", "2020-04"],
        "remaining_lease": ["99 years", "95 years", "90 years", "85 years"],
        "price": [300_000, 320_000, 340_000, 360_000],
        # plus dummy columns to drop
        "id":[1,2,3,4],"block":["A"]*4,"street_name":["X"]*4
    })

def test_end_to_end_training(raw_df, full_config):
    dp = DataPreparation(full_config)
    cleaned = dp.clean_data(raw_df)
    mt = ModelTraining(full_config, dp.preprocessor)
    X_train, X_val, X_test, y_train, y_val, y_test = mt.split_data(cleaned)
    # train baseline only
    pipelines, metrics = mt.train_and_evaluate_baseline_models(X_train, y_train, X_val, y_val)
    # we expect one metric dict per model
    assert set(pipelines.keys()) == {"Decision_Tree_Baseline", "Random_Forest_Baseline", "XGBoost_Baseline", "LightGBM_Baseline"}
    for name, m in metrics.items():
        assert all(k in m for k in ["MAE","MSE","RMSE","R²"])
    # final evaluation on test set works without exceptions
    final = mt.evaluate_final_model(next(iter(pipelines.values())), X_test, y_test, "any")
    assert "MAE" in final
