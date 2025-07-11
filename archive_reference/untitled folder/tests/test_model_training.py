import pandas as pd
import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from model_training import ModelTraining

@pytest.fixture
def tiny_df():
    # numeric + categorical + target
    return pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0],
        "cat": ["A", "B", "A", "B"],
        "pass": [8, 11, 5, 2],
        "price": [10.0, 20.0, 30.0, 40.0]
    })

@pytest.fixture
def dummy_preprocessor():
    # passthrough numeric, one-hot encode cat
    return ColumnTransformer([
        ("num",  "passthrough", ["num"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["cat"]),
        ("pass",  "passthrough", ['pass'])
    ], remainder="passthrough")

@pytest.fixture
def config():
    return {
        "target_column": "price",
        "val_test_size": 0.5,
        "val_size": 0.5
    }

def test_split_data_shapes(tiny_df, dummy_preprocessor, config):
    mt = ModelTraining(config, dummy_preprocessor)
    X_train, X_val, X_test, y_train, y_val, y_test = mt.split_data(tiny_df)
    # total rows = 4; train 2, val 1, test 1
    assert X_train.shape[0] == 2
    assert X_val.shape[0] == 1
    assert X_test.shape[0] == 1
    # features dropped target column
    assert "price" not in X_train.columns
    # targets are Series of length matching
    assert len(y_train) == 2

def test_evaluate_model_metrics(tiny_df, dummy_preprocessor, config):
    # train a trivial model: regressor that always predicts mean
    from sklearn.dummy import DummyRegressor
    mt = ModelTraining(config, dummy_preprocessor)
    X_train, X_val, _, y_train, y_val, _ = mt.split_data(tiny_df)
    pipeline = dummy_preprocessor
    # wrap DummyRegressor in pipeline
    from sklearn.pipeline import Pipeline
    model = Pipeline([("preprocessor", dummy_preprocessor), ("regressor", DummyRegressor())])
    model.fit(X_train, y_train)
    metrics = mt._evaluate_model(model, X_val, y_val, "dummy")
    # for DummyRegressor, MAE should be non-negative
    assert metrics["MAE"] >= 0
    assert "R²" in metrics

@pytest.mark.parametrize("empty_df", [pd.DataFrame(), pd.DataFrame(columns=["num","cat","price"])])
def test_split_data_empty(empty_df, dummy_preprocessor, config):
    mt = ModelTraining(config, dummy_preprocessor)
    with pytest.raises(ValueError) as exc:
        mt.split_data(empty_df)
    assert "empty" in str(exc.value).lower()