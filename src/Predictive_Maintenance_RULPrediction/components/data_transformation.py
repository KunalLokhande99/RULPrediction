import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from src.Predictive_Maintenance_RULPrediction.logger import logging
from src.Predictive_Maintenance_RULPrediction.exception import CustomException
from src.Predictive_Maintenance_RULPrediction.utils import save_object
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    filename='data_transformation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)
logger = logging.getLogger()

@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.artifacts_dir = os.path.join(os.getcwd(), 'artifacts')
        self.preprocessor_obj_file_path = os.path.join(self.artifacts_dir, 'preprocessor.pkl')
        self.transformed_train_path = os.path.join(self.artifacts_dir, 'transformed_train.csv')
        self.transformed_test_path = os.path.join(self.artifacts_dir, 'transformed_test.csv')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        logger.info("DataTransformation initialized")

    def _verify_input_files(self, *paths):
        """Verify that input files exist"""
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Input file not found: {path}")
            logger.info(f"Verified input file exists: {path}")

    def initiate_data_transformation(self, train_path, test_path, rul_path):
        logger.info("Starting data transformation process")
        print("=== Starting Data Transformation ===")

        try:
            # 1. Verify input files exist
            print("\n[1/4] Verifying input files...")
            self._verify_input_files(train_path, test_path, rul_path)
            print("✓ Input files verified")

            # 2. Create artifacts directory if not exists
            print("\n[2/4] Ensuring artifacts directory exists...")
            os.makedirs(self.data_transformation_config.artifacts_dir, exist_ok=True)
            if not os.path.exists(self.data_transformation_config.artifacts_dir):
                raise RuntimeError("Failed to create artifacts directory")
            print(f"✓ Directory verified: {self.data_transformation_config.artifacts_dir}")

            # 3. Process data files
            print("\n[3/4] Processing data files...")
            
            # Reading train, test, and rul data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            rul_df = pd.read_csv(rul_path)
            logger.info('Read train, test and RUL data completed')

            # Defining Column Names for the dataset

            column_names = ['unit', 'time', 'operational_setting_1', 'operational_setting_2', 'operational_setting_3',
               'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8',
               'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16',
               'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21']
            rul_names = ['RUL']

            train_df.columns=column_names
            test_df.columns=column_names
            rul_df.columns=rul_names

            # Dropping columns for better model performance
            columns_to_drop = ['operational_setting_3','sensor_1','sensor_5','sensor_6',
                             'sensor_10','sensor_16','sensor_18','sensor_19']
            train_df = train_df.drop(columns=columns_to_drop, axis=1)
            test_df = test_df.drop(columns=columns_to_drop, axis=1)

            # Calculating RUL for train and test data
            train_df['RUL'] = train_df.groupby('unit')['time'].transform('max') - train_df['time']
            test_df['RUL'] = test_df.groupby('unit')['time'].transform('max') - test_df['time']
            rul_df['unit'] = rul_df.index + 1
            test_df = pd.merge(test_df, rul_df, on='unit', how='left')
            test_df['RUL'] = test_df['RUL_x'] + test_df['RUL_y']
            test_df = test_df.drop(columns=['RUL_x', 'RUL_y'], axis=1)

            # Capping RUL at 25th percentile (51)
            train_df["RUL"][train_df["RUL"] > 51] = 51
            test_df["RUL"][test_df["RUL"] > 51] = 51

            logger.info(f'Train Dataframe head: \n{train_df.head().to_string()}')
            logger.info(f'Test Dataframe head: \n{test_df.head().to_string()}')

            # Splitting data into input and target features
            target_column = 'RUL'
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            # Transforming using RobustScaler
            scaler = RobustScaler()
            input_feature_train_arr = scaler.fit_transform(input_feature_train_df)
            input_feature_test_arr = scaler.transform(input_feature_test_df)
            logger.info("Feature scaling completed")

            # Combining features and targets
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # 4. Save outputs and verify
            print("\n[4/4] Saving transformed data...")
            
            # Save preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=scaler
            )
            
            # Save transformed data as CSV
            pd.DataFrame(train_arr).to_csv(self.data_transformation_config.transformed_train_path, index=False)
            pd.DataFrame(test_arr).to_csv(self.data_transformation_config.transformed_test_path, index=False)
            
            # Verify outputs
            outputs = {
                'preprocessor': self.data_transformation_config.preprocessor_obj_file_path,
                'transformed_train': self.data_transformation_config.transformed_train_path,
                'transformed_test': self.data_transformation_config.transformed_test_path
            }
            
            for name, path in outputs.items():
                if not os.path.exists(path):
                    raise RuntimeError(f"Failed to create output file: {path}")
                print(f"✓ {name} saved to {path}")
                logger.info(f"{name} saved successfully at {path}")

            print("\n=== Data Transformation Completed Successfully ===")
            logger.info("Data transformation completed successfully")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            error_msg = f"Data transformation failed: {str(e)}"
            logger.error(error_msg)
            print(f"\n!!! ERROR: {error_msg}")
            raise CustomException(e, sys)

# Example usage
if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    print(f"Log file will be created at: {os.path.abspath('data_transformation.log')}")
    
    try:
        # Example paths (replace with actual paths from DataIngestion)
        train_path = "artifacts/train.csv"
        test_path = "artifacts/test.csv"
        rul_path = "artifacts/rul.csv"

        transformer = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(
            train_path, test_path, rul_path
        )
        
        print("\nFinal Outputs:")
        print(f"Train array shape: {train_arr.shape}")
        print(f"Test array shape: {test_arr.shape}")
        print(f"Preprocessor path: {preprocessor_path}")
        
    except Exception as e:
        print(f"\nFatal error occurred: {str(e)}")
        sys.exit(1)

    
    