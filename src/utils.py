import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exeption import CustomException
from src.logger import logger

from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logger.error('Exception occured during model saving...')
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {'Model_Name':[], 'Model': [], 'R2_Score': []}

        for i in range(len(models)):
            model = list(models.values())[i]
            # model_name = list(models.keys())[i]

            logger.info(f'Training on {model}')

            # Train model
            model.fit(X_train,y_train)

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            logger.info(f'R2_Score: {test_model_score}')

            report['Model_Name'].append(list(models.keys())[i])
            report['Model'].append(model)
            report['R2_Score'].append(test_model_score*100)

        return report

    except Exception as e:
        logger.error('Exception occured during model training')
        raise CustomException(e,sys)

def enhance_model(X_train,y_train,X_test,y_test,models, params):
    try:
        report = {'Model_Name':[], 'Model': [], 'R2_Score': []}

        model = list(models.values())[0]
        # model_name = list(models.keys())[i]

        logger.info(f'Enhancing {model}')

        grid_search=GridSearchCV(estimator=model,param_grid=params,cv=5, verbose=10)
        grid_search.fit(X_train,y_train)

        logger.info(f'Best Estimator: {grid_search.best_estimator_}')
        logger.info(f'Best Param: {grid_search.best_params_}')

        model.set_params(**grid_search.best_params_)

        # Train model
        model.fit(X_train,y_train)

        # Predict Testing data
        y_test_pred =model.predict(X_test)

        # Get R2 scores for train and test data
        #train_model_score = r2_score(ytrain,y_train_pred)
        test_model_score = r2_score(y_test,y_test_pred)
        logger.info(f'R2_Score: {test_model_score}')

        report['Model_Name'].append(list(models.keys())[0])
        report['Model'].append(model)
        report['R2_Score'].append(test_model_score*100)

        return report

    except Exception as e:
        logger.error('Exception occured during model training')
        raise CustomException(e,sys)

    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logger.error('Exception Occured in load_object function utils')
        raise CustomException(e,sys)

    