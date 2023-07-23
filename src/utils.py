import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exeption import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.error('Exception occured during model saving...')
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {'Model_Name':[], 'Model': [], 'R2_Score': []}

        for i in range(len(models)):
            model = list(models.values())[i]
            # model_name = list(models.keys())[i]

            logging.info(f'Training on {model}')

            # Train model
            model.fit(X_train,y_train)

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            logging.info(f'R2_Score: {test_model_score}')

            report['Model_Name'].append(list(models.keys())[i])
            report['Model'].append(model)
            report['R2_Score'].append(test_model_score*100)

        return report

    except Exception as e:
        logging.error('Exception occured during model training')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.error('Exception Occured in load_object function utils')
        raise CustomException(e,sys)

    