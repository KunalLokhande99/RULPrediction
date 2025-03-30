import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from src.Predictive_Maintenance_RULPrediction.logger import logging
from src.Predictive_Maintenance_RULPrediction.exception import CustomException
from src.Predictive_Maintenance_RULPrediction.utils import save_object, evaluate_models

# Configure logging
logging.basicConfig(
    filename='model_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)
logger = logging.getLogger()

@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.artifacts_dir = os.path.join(os.getcwd(), 'artifacts')
        self.trained_model_file_path = os.path.join(self.artifacts_dir, 'model.pkl') #use model.pkl

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        logger.info("ModelTrainer initialized")

    def _load_and_validate_data(self, train_path, test_path):
        """Load and validate input data"""
        try:
            # Load data using pandas to handle CSV formatting
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Convert to numpy arrays
            train_array = train_df.values
            test_array = test_df.values
            
            # Verify data shape and type
            if train_array.shape[1] < 2 or test_array.shape[1] < 2:
                raise ValueError("Input arrays must have at least 2 columns (features + target)")
                
            # Convert to float and verify no string values
            train_array = train_array.astype(float)
            test_array = test_array.astype(float)
            
            logger.info("Data loaded and validated successfully")
            return train_array, test_array
            
        except Exception as e:
            error_msg = f"Data loading failed: {str(e)}"
            logger.error(error_msg)
            raise CustomException(error_msg, sys)

    def initiate_model_training(self, train_path, test_path):
        logger.info("Starting model training process")
        print("=== Starting Model Training ===")

        try:
            # 1. Load and verify input data
            print("\n[1/4] Loading and validating data...")
            train_array, test_array = self._load_and_validate_data(train_path, test_path)
            print("✓ Data loaded and validated")

            # 2. Ensure artifacts directory exists
            print("\n[2/4] Ensuring artifacts directory exists...")
            os.makedirs(self.model_trainer_config.artifacts_dir, exist_ok=True)
            if not os.path.exists(self.model_trainer_config.artifacts_dir):
                raise RuntimeError("Failed to create artifacts directory")
            print(f"✓ Directory verified: {self.model_trainer_config.artifacts_dir}")

            # 3. Train and evaluate models
            print("\n[3/4] Training and evaluating models...")
            
            # Applying train test split
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            # Selecting models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "SupportVector Regressor": SVR(),
            }

            # Defining parameters for models
            params = {
                'Random Forest': {
                    'n_estimators': [200],
                    'max_depth': [20],
                    'min_samples_split': [5],
                    'min_samples_leaf': [2]
                },
                'Gradient Boosting': {
                    'n_estimators': [200],
                    'learning_rate': [0.05],
                    'max_depth': [20],
                    'min_samples_split': [5],
                    'min_samples_leaf': [2]
                },
                'XGBRegressor': {
                    'n_estimators': [200],
                    'learning_rate': [0.05],
                    'max_depth': [20],
                    'min_child_weight': [5]
                },
                'SupportVector Regressor': {
                    'C': [5],
                    'kernel': ['rbf'],
                    'degree': [2],
                    'epsilon': [0.01,]
                },
            }
            
            model_report = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, params=params
            )

            # Get best model name and score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(f'Best Model Found: {best_model_name}, R2 Score: {best_model_score}')
            logger.info(f'Best model: {best_model_name} with score: {best_model_score}')

            if best_model_score < 0.6:
                warning_msg = "No best model found with R2 score >= 0.6"
                print(warning_msg)
                logger.warning(warning_msg)

            # 4. Save model
            print("\n[4/4] Saving best model...")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            if not os.path.exists(self.model_trainer_config.trained_model_file_path):
                raise RuntimeError(f"Failed to create model file: {self.model_trainer_config.trained_model_file_path}")
            print(f"✓ Model saved to {self.model_trainer_config.trained_model_file_path}")
            logger.info(f"Model saved successfully at {self.model_trainer_config.trained_model_file_path}")


            print("\n=== Model Training Completed Successfully ===")
            logger.info("Model training completed successfully")
            
            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            logger.error(error_msg)
            print(f"\n!!! ERROR: {error_msg}")
            raise CustomException(e, sys)

# Example usage
if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    print(f"Log file will be created at: {os.path.abspath('model_training.log')}")
    
    try:
        # Example paths (replace with actual transformed data paths)
        train_path = "artifacts/transformed_train.csv"
        test_path = "artifacts/transformed_test.csv"

        trainer = ModelTrainer()
        model_path = trainer.initiate_model_training(train_path, test_path)
        
        print("\nFinal Output:")
        print(f"Model path: {model_path}")
        
    except Exception as e:
        print(f"\nFatal error occurred: {str(e)}")
        sys.exit(1)