import os
import sys
import numpy as np
import pandas as pd
from src.Predictive_Maintenance_RULPrediction.components.data_ingestion import DataIngestion
from src.Predictive_Maintenance_RULPrediction.components.data_transformation import DataTransformation
from src.Predictive_Maintenance_RULPrediction.components.model_trainer import ModelTrainer
from src.Predictive_Maintenance_RULPrediction.logger import logging
from src.Predictive_Maintenance_RULPrediction.exception import CustomException

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        
    def run_pipeline(self):
        """Execute the complete training pipeline"""
        try:
            logging.info("Starting training pipeline")
            print("\n" + "="*50)
            print("Starting Predictive Maintenance Training Pipeline")
            print("="*50 + "\n")
            
            # 1. Data Ingestion
            print("\n[1/3] Running Data Ingestion...")
            train_path, test_path, rul_path = self.data_ingestion.initiate_data_ingestion()
            print("✓ Data Ingestion Completed")
            print(f"Train data: {train_path}")
            print(f"Test data: {test_path}")
            print(f"RUL data: {rul_path}")
            
            # 2. Data Transformation
            print("\n[2/3] Running Data Transformation...")
            train_arr, test_arr, preprocessor_path = self.data_transformation.initiate_data_transformation(
                train_path, test_path, rul_path
            )
            print("✓ Data Transformation Completed")
            print(f"Train array shape: {train_arr.shape}")
            print(f"Test array shape: {test_arr.shape}")
            print(f"Preprocessor path: {preprocessor_path}")
            
            # Save transformed arrays as CSV files for ModelTrainer
            train_arr_path = os.path.join("artifacts", "transformed_train.csv")
            test_arr_path = os.path.join("artifacts", "transformed_test.csv")
            
            # Save with header for better readability
            pd.DataFrame(train_arr).to_csv(train_arr_path, index=False, header=False)
            pd.DataFrame(test_arr).to_csv(test_arr_path, index=False, header=False)
            print(f"Saved transformed train data to: {train_arr_path}")
            print(f"Saved transformed test data to: {test_arr_path}")
            
            # 3. Model Training (using file paths)
            print("\n[3/3] Running Model Training...")
            model_path = self.model_trainer.initiate_model_training(train_arr_path, test_arr_path)
            print("✓ Model Training Completed")
            print(f"Model saved at: {model_path}")
            
            print("\n" + "="*50)
            print("Training Pipeline Completed Successfully!")
            print("="*50 + "\n")
            logging.info("Training pipeline completed successfully")
            
        except Exception as e:
            error_msg = f"Training pipeline failed: {str(e)}"
            logging.error(error_msg)
            print(f"\n!!! ERROR: {error_msg}")
            raise CustomException(error_msg, sys)

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        print(f"\nPipeline execution failed: {str(e)}")
        sys.exit(1)

        