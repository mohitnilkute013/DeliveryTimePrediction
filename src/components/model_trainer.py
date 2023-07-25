# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.svm import SVR
import xgboost as xgb

from src.exeption import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet(),
            'DecisionTree':DecisionTreeRegressor(random_state=42),
            'SVR linear':SVR(kernel='linear'),
            'SVR rbf':SVR(kernel='rbf'),
            'KNNR':KNeighborsRegressor(n_neighbors=3),
            'RandomForest':RandomForestRegressor(random_state=42),
            'AdaBoost':AdaBoostRegressor(estimator=DecisionTreeRegressor(random_state=42)),
            'Gradient Boosting':GradientBoostingRegressor(),
            'XGB':xgb.XGBRegressor(),
            'BaggingSVR':BaggingRegressor(estimator=SVR())
            }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(pd.DataFrame(model_report))
            print('\n====================================================================================\n')
            logging.info(f'Model Report : \n{pd.DataFrame(model_report)}')

            # To get best model score from dictionary 
            index = list(model_report['R2_Score']).index(max(model_report['R2_Score']))

            best_model_score = model_report['R2_Score'][index]
            best_model_name = model_report['Model_Name'][index]
            best_model = model_report['Model'][index]


            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )

            logging.info('Saved Best Model file')


        except Exception as e:
            logging.error('Exception occured at Model Training')
            raise CustomException(e,sys)