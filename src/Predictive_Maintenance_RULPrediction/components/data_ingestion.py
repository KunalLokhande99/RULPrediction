import pandas as pd
import numpy as np
from src.Predictive_Maintenance_RULPrediction.logger import logging
from src.Predictive_Maintenance_RULPrediction.exception import CustomException
import mlflow
import mlflow.sklearn
logging.basicConfig(filename='data_ingestion.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='data_ingestion.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # Overwrite existing log file
)
logger = logging.getLogger()

class DataIngestionConfig:
    def __init__(self):
        self.artifacts_dir = os.path.join(os.getcwd(), 'artifacts')
        self.train_data_path = os.path.join(self.artifacts_dir, 'train.csv')
        self.test_data_path = os.path.join(self.artifacts_dir, 'test.csv')
        self.rul_data_path = os.path.join(self.artifacts_dir, 'rul.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        logger.info("DataIngestion initialized")

    def _verify_input_files(self, *paths):
        """Verify that input files exist"""
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Input file not found: {path}")
            logger.info(f"Verified input file exists: {path}")

    def initiate_data_ingestion(self):
        logger.info("Starting data ingestion process")
        print("=== Starting Data Ingestion ===")

        try:
            # Define input paths (using raw strings for Windows paths)
            input_paths = {
                'train': r'C:\RULPrediction\notebooks\data\train_FD001.txt',
                'test': r'C:\RULPrediction\notebooks\data\test_FD001.txt',
                'rul': r'C:\RULPrediction\notebooks\data\rul_FD001.txt'
            }

            # 1. Verify input files exist
            print("\n[1/4] Verifying input files...")
            self._verify_input_files(*input_paths.values())
            print("✓ Input files verified")

            # 2. Create artifacts directory
            print("\n[2/4] Creating artifacts directory...")
            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)
            if not os.path.exists(self.ingestion_config.artifacts_dir):
                raise RuntimeError("Failed to create artifacts directory")
            print(f"✓ Directory created: {self.ingestion_config.artifacts_dir}")

            # 3. Read and save data
            print("\n[3/4] Processing data files...")
            results = {}
            for name, path in input_paths.items():
                print(f"Processing {name} data...")
                df = pd.read_csv(path, sep='\s+', header=None)
                output_path = getattr(self.ingestion_config, f"{name}_data_path")
                df.to_csv(output_path, index=False)
                
                # Verify output
                if not os.path.exists(output_path):
                    raise RuntimeError(f"Failed to create output file: {output_path}")
                
                results[name] = {
                    'input_path': path,
                    'output_path': output_path,
                    'shape': df.shape
                }
                print(f"✓ {name} data saved to {output_path}")

            # 4. Final verification
            print("\n[4/4] Verifying outputs...")
            for name, data in results.items():
                print(f"{name}: {data['shape']} -> {data['output_path']}")
            print("✓ All outputs verified")

            print("\n=== Data Ingestion Completed Successfully ===")
            logger.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.rul_data_path
            )

        except Exception as e:
            error_msg = f"Data ingestion failed: {str(e)}"
            logger.error(error_msg)
            print(f"\n!!! ERROR: {error_msg}")
            raise

# Example usage
if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    print(f"Log file will be created at: {os.path.abspath('data_ingestion.log')}")
    
    try:
        ingestion = DataIngestion()
        train_path, test_path, rul_path = ingestion.initiate_data_ingestion()
        
        print("\nFinal Output Paths:")
        print(f"Train: {train_path}")
        print(f"Test: {test_path}")
        print(f"RUL: {rul_path}")
        
    except Exception as e:
        print(f"\nFatal error occurred: {str(e)}")
        sys.exit(1)