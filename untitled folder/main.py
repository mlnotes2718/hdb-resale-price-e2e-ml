# Standard library imports
import logging
from sklearn.utils._testing import ignore_warnings
import warnings


# Third-party imports
import pandas as pd
import yaml
import joblib
from sklearn.utils._testing import ignore_warnings

# Local application/library specific imports
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining
from src.optuna_params import params_config

logging.basicConfig(level=logging.INFO)


@ignore_warnings(category=UserWarning) # type: ignore
def main():

    # Configuration file path
    config_path = "./src/config.yaml"

    # Load configuration
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        logging.info(f'Exception (load configuration): {e}')

    # Load CSV file into a DataFrame
    try:
        df = pd.read_csv(config["file_path"])
    except Exception as e:
        logging.info(f'Exception (load raw data): {e}')

    # Initialize and run data preparation
    try:
        data_prep = DataPreparation(config)
    except Exception as e:
        logging.info(f'Exception (run data preparation): {e}')

    try:
        cleaned_df = data_prep.clean_data(df)
    except Exception as e:
        logging.info(f'Exception (data cleaning): {e}')


    # Initialize model training with the created preprocessor
    try:
        model_training = ModelTraining(config, data_prep.preprocessor)
    except Exception as e:
        logging.info(f'Exception (initialize model training class): {e}')
    
    # Split the data
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = model_training.split_data(
            cleaned_df
        )
    except Exception as e:
        logging.info(f'Exception (train test split): {e}')    

    # Train and evaluate baseline models with default hyperparameters
    try:
        baseline_models, baseline_metrics = (
            model_training.train_and_evaluate_baseline_models(
                X_train, y_train, X_val, y_val
            )
        )
    except Exception as e:
        logging.info(f'Exception (training base line model): {e}')   

    # Train and evaluate tuned models with hyperparameter tuning
    try:
        tuned_models, tuned_metrics = model_training.train_and_evaluate_tuned_models(
            X_train, y_train, X_val, y_val, params_config=params_config
        )
    except Exception as e:
        logging.info(f'Exception (hyperparameter tuning): {e}')   

    # Combine all models and their metrics into dictionaries
    all_models = {**baseline_models, **tuned_models}
    all_metrics = {**baseline_metrics, **tuned_metrics}

    # Find the best model based on MAE score
    best_model_name = min(all_metrics, key=lambda k: all_metrics[k]["MAE"])
    best_model = all_models[best_model_name]
    logging.info(f"Best Model Found: {best_model_name}")

    # Evaluate the best model on the test set
    final_metrics = model_training.evaluate_final_model(
        best_model, X_test, y_test, best_model_name
    )

    # Save model to joblib file
    # Define the filename
    hdb_resale_model_joblib = 'hdb_resale_best_model.joblib'

    # Save the model using joblib
    try:
        joblib.dump(best_model, hdb_resale_model_joblib)
        logging.info(f"Model successfully saved to {hdb_resale_model_joblib} using joblib.")
    except Exception as e:
        print(f"Error saving model with joblib: {e}")


if __name__ == "__main__":
    main()
