import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,mean_absolute_error
from src.Predictive_Maintenance_RULPrediction.logger import logging
from src.Predictive_Maintenance_RULPrediction.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        logging.info('Exception occurred in save_object function utils')
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occurred in load_object function utils')
        raise CustomException(e, sys)
    
    
# evaluating models performance

def evaluate_models(X_train, y_train,X_test,y_test,models,params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]

            gscv = GridSearchCV(model,para,cv=3,n_jobs=-1)
            gscv.fit(X_train,y_train)

            model.set_params(**gscv.best_params_)
            model.fit(X_train,y_train)

            y_pred = model.predict(X_test)

            model_score = r2_score(y_test, y_pred)
            report[list(models.keys())[i]] = model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)