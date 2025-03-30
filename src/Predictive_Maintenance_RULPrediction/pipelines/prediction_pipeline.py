import os
import sys
import pandas as pd
from src.Predictive_Maintenance_RULPrediction.logger import logging
from src.Predictive_Maintenance_RULPrediction.exception import CustomException
from src.Predictive_Maintenance_RULPrediction.utils import load_object

class PredictPipeline:
    def __init__(self):
        # Separate paths for model and preprocessor
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")  
        
    def predict(self, features):
        """Make predictions on new data"""
        try:
            logging.info("Loading model and preprocessor")
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)
            
            # scale the features
            data_scaled = preprocessor.transform(features)
            
            logging.info("Making predictions")
            preds = model.predict(data_scaled)
            
            return preds
            
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 unit: float,
                 time: float,
                 operational_setting_1: float,
                 operational_setting_2: float,
                 sensor_2: float,
                 sensor_3: float,
                 sensor_4: float,
                 sensor_7: float,
                 sensor_8: float,
                 sensor_9: float,
                 sensor_11: float,
                 sensor_12: float,
                 sensor_13: float,
                 sensor_14: float,
                 sensor_15: float,
                 sensor_17: float,
                 sensor_20: float,
                 sensor_21: float):
        
        self.unit = unit
        self.time = time
        self.operational_setting_1 = operational_setting_1
        self.operational_setting_2 = operational_setting_2
        self.sensor_2 = sensor_2
        self.sensor_3 = sensor_3
        self.sensor_4 = sensor_4
        self.sensor_7 = sensor_7
        self.sensor_8 = sensor_8
        self.sensor_9 = sensor_9
        self.sensor_11 = sensor_11
        self.sensor_12 = sensor_12
        self.sensor_13 = sensor_13
        self.sensor_14 = sensor_14
        self.sensor_15 = sensor_15
        self.sensor_17 = sensor_17
        self.sensor_20 = sensor_20
        self.sensor_21 = sensor_21
    
    def get_data_as_dataframe(self):
        """Convert input data to DataFrame with correct column structure"""
        try:
            custom_data_dict = {
                "unit": [self.unit],
                "time": [self.time],
                "operational_setting_1": [self.operational_setting_1],
                "operational_setting_2": [self.operational_setting_2],
                "sensor_2": [self.sensor_2],
                "sensor_3": [self.sensor_3],
                "sensor_4": [self.sensor_4],
                "sensor_7": [self.sensor_7],
                "sensor_8": [self.sensor_8],
                "sensor_9": [self.sensor_9],
                "sensor_11": [self.sensor_11],
                "sensor_12": [self.sensor_12],
                "sensor_13": [self.sensor_13],
                "sensor_14": [self.sensor_14],
                "sensor_15": [self.sensor_15],
                "sensor_17": [self.sensor_17],
                "sensor_20": [self.sensor_20],
                "sensor_21": [self.sensor_21]
            }
            
            print("\n" + "="*50)
            print("Prediction Pipeline Completed Successfully!")
            print("="*50 + "\n")
            logging.info("Prediction pipeline completed successfully")
            
            return pd.DataFrame(custom_data_dict)
        
            
        except Exception as e:
            logging.error(f"Data conversion failed: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Sample input data
        input_data = CustomData(
            unit=1,
            time=150,
            operational_setting_1=25.0,
            operational_setting_2=0.52,
            sensor_2=518.67,
            sensor_3=641.82,
            sensor_4=1589.70,
            sensor_7=554.36,
            sensor_8=2388.06,
            sensor_9=9046.19,
            sensor_11=47.47,
            sensor_12=521.66,
            sensor_13=2388.02,
            sensor_14=8138.62,
            sensor_15=8.4195,
            sensor_17=392.20,
            sensor_20=39.06,
            sensor_21=23.4190
        )
        
        # Convert to DataFrame
        features = input_data.get_data_as_dataframe()
        
        # Make prediction
        pipeline = PredictPipeline()
        predictions = pipeline.predict(features)
        
        print(f"\nPredicted RUL: {predictions[0]:.2f} cycles")
        
    except Exception as e:
        print(f"\nError occurred during prediction: {str(e)}")
        sys.exit(1)

