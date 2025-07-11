import pandas as pd
import pytest
from data_preparation import DataPreparation

@pytest.fixture
def sample_df():
    # minimal DataFrame with the columns used in clean_data()
    return pd.DataFrame({
        "lease_commence_date": [-10, 20],
        "flat_type": ["FOUR ROOM", "5 ROOM"],
        "town_id": [1, 2],
        "town_name": [None, "TOWN B"],
        "flatm_id": [10, 20],
        "flatm_name": ["TYPE A", None],
        "storey_range": ["07 TO 09", "10 TO 12"],
        "month": ["2020-01", "2021-12"],
        "remaining_lease": ["99 years 6 months", "80 years"],
        "id": [100, 101],
        "block": ["A", "B"],
        "street_name": ["X St", "Y Ave"]
    })

@pytest.fixture
def config():
    return {
        "numerical_features": ["transac_year",'remaining_lease_by_months'],
        "nominal_features": ["flat_type"],
        "passthrough_features": ['storey_range']
    }

def test_convert_storey_range():
    assert DataPreparation._convert_storey_range("07 TO 09") == 8.0
    assert DataPreparation._convert_storey_range("10 TO 12") == 11.0

def test_convert_lease_to_month():
    assert DataPreparation._convert_lease_to_month("1 years 2 months") == 14
    assert DataPreparation._convert_lease_to_month("5 years") == 60
    assert DataPreparation._convert_lease_to_month("0 years 3 months") == 3

def test_handling_missing_name(sample_df: pd.DataFrame):
    # fill town_name from town_id mapping
    df = sample_df.copy()
    # intentionally create mapping: town_id=2 maps to "TOWN B"
    df.loc[0, "town_id"] = 2
    df = DataPreparation._handling_missing_name(df, "town_name", "town_id")
    assert df.loc[0, "town_name"] == "TOWN B"

def test_clean_data(sample_df: pd.DataFrame, config: dict[str, list[str]]):
    prep = DataPreparation(config)
    cleaned = prep.clean_data(sample_df.copy())

    # No duplicates
    assert cleaned.shape[0] == 2

    # lease_commence_date is positive
    assert (cleaned.remaining_lease_by_months >= 0).all()

    # flat_type replaced
    assert "4 ROOM" in cleaned["flat_type"].values

    # new columns exist
    assert set(["transac_year", "transac_month", "remaining_lease_by_months"]).issubset(cleaned.columns)

    # dropped columns
    for col in ["id", "month", "block", "street_name"]:
        assert col not in cleaned.columns
