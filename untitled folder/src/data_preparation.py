# Import Python Modules

# Essential Modules
import pandas as pd
import logging
from typing import Any, Dict

# Pipeline and Train Test Split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# SciKit Learning Preprocessing  
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures


# Data Preparation Class
class DataPreparation:
    """
    A class used to clean and preprocess HDB resale prices data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    """

    # Constructor
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparation class with a configuration dictionary.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for data cleaning and preprocessing.
        """
        self.config = config
        self.preprocessor = self._create_preprocessor()

    # Data Cleaning Method
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by performing several preprocessing steps.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the raw data.

        Returns:
        --------
        pd.DataFrame: The cleaned DataFrame.
        """
        # Logging
        logging.info("Starting data cleaning.")

        ## Data Cleaning
        # drop duplicates
        df.drop_duplicates(inplace=True)

        # Remove negative number in lease commence date
        df.lease_commence_date = df.lease_commence_date.abs()

        # Rename flat type to 4 ROOM
        df.flat_type = df.flat_type.replace('FOUR ROOM', '4 ROOM')

        # Missing Values
        df = self._handling_missing_name(df, missing_name_col='town_name', missing_name_related_id_col='town_id')
        df = self._handling_missing_name(df=df, missing_name_related_id_col='flatm_id', missing_name_col='flatm_name')

        ## Feature Engineering
        # Convert storey range to number
        df.storey_range = df.storey_range.apply(self._convert_storey_range)

        # Convert to year and month
        df['year_month'] = pd.to_datetime(df.month, format='%Y-%m')
        df['transac_year'] = df.year_month.dt.year
        df['transac_month'] = df.year_month.dt.month


        # Convert remaining lease to months
        df['remaining_lease_by_months'] = df.remaining_lease.apply(self._convert_lease_to_month)

        # Dropping columns
        df.drop(columns=['id', 'month', 'block', 'street_name', 'remaining_lease', 'town_id',  'flatm_id', 'year_month', 'lease_commence_date'], inplace=True)
        return df

    # Create Preprocessor Method
    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor pipeline for transforming numerical, nominal, and ordinal features.

        Returns:
        --------
        sklearn.compose.ColumnTransformer: A ColumnTransformer object for preprocessing the data.
        """
        nominal_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", self.config["numerical_features"]),
                ("nom", nominal_transformer, self.config["nominal_features"]),
                ("pass", "passthrough", self.config["passthrough_features"]),
            ],
            remainder="passthrough",
            n_jobs=-1,
        )
        return preprocessor




    @staticmethod
    def _convert_storey_range(storey_range: str) -> float:
        """
        Convert storey range to the numerical average.
        Args:
            storey_range in (str)

        Returns: 
            float

        Example:
            convert_storey_range('07 TO 09') -> 8.0 
        """

        low, high = storey_range.split(' TO ')
        average = (int(low) + int(high)) / 2
        return average

    @staticmethod
    def _handling_missing_name(df: pd.DataFrame, missing_name_col: str, missing_name_related_id_col: str) -> pd.DataFrame:
        """
        Fills missing values in the 'name' column from the 'id' column.

        Args:
        -----
        df (pd.DataFrame): The DataFrame containing the columns to be fixed.
        missing_name_col (str): The name of the column containing missing names to be filled.
        missing_name_related_id_col (str): The name of the column containing the IDs that matches the name.

        Returns:
        --------
        pd.DataFrame: The DataFrame with missing values fixed.
        """
        missing_name_rows = df[missing_name_col].isnull()
        list_name = df[missing_name_col].value_counts().index.to_list()
        list_id = df[missing_name_related_id_col].value_counts().index.to_list()

        missing_name_mapping = dict(zip(list_id, list_name))
        #print(missing_name_mapping)
        
        df.loc[missing_name_rows, missing_name_col] = df.loc[missing_name_rows, missing_name_related_id_col].map(missing_name_mapping)

        return df

    @staticmethod
    def _convert_lease_to_month(lease: str) -> int:
        """
        Convert remaining lease period from string to total number of months.
        Args:
            lease in (str)

        Returns: 
            integer

        Example:
            convert_lease_to_month('07 TO 09') -> 8.0  
        """
        str_list = lease.split(' ')
        if ('months' in str_list) | ('month' in str_list):
            year = int(str_list[0])
            month = int(str_list[2])
            total_month = (year * 12) + month 
        elif ('years' in str_list) & (('months' not in str_list) | ('month' not in str_list)):
            year = int(str_list[0])
            total_month = (year * 12)
        else:
            year = int(str_list[0])
            total_month = (year * 12)        
        return total_month

