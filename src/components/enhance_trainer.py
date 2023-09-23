import os
import sys

import pandas as pd 
import numpy as np 

from src.logger import logger
from src.exeption import CustomException
from src.utils import save_object, load_object, evaluate_model, enhance_model

from dataclasses import dataclass

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

import xgboost as xgb


@dataclass
class EnhanceTrainerConfig:
    enhance_trainer_config_path = os.path.join(os.getcwd(), 'artifacts', 'enhanced_model.pkl')

class EnhanceTrainer:

    def __init__(self):
        self._config = EnhanceTrainerConfig()
    
    def initiate_training(self, trainarr, testarr):
        try:
            models = {
                'XGB':xgb.XGBRegressor()
            }

            logger.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                trainarr[:,:-1],
                trainarr[:,-1],
                testarr[:,:-1],
                testarr[:,-1]
            )

            param_grid = {
                'eta': [0.01],
                'gamma': [1, 2, 3],
                'max_depth': [3, 4, 5, 6, 7],
                'min_child_weight':[1, 2, 3],
                }

            model_report = enhance_model(X_train, y_train, X_test, y_test, models, params = param_grid)

            print(pd.DataFrame(model_report))
            print('\n====================================================================================\n')
            logger.info(f'Model Report : \n{pd.DataFrame(model_report)}')

            # To get best model score from dictionary 
            index = list(model_report['R2_Score']).index(max(model_report['R2_Score']))

            best_model_score = model_report['R2_Score'][index]
            best_model_name = model_report['Model_Name'][index]
            best_model = model_report['Model'][index]


            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logger.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self._config.enhance_trainer_config_path,
                 obj=best_model
            )

            logger.info('Saved Best Model file')

        except Exception as e:
            logger.error('Error in Enhance Training.')
            raise CustomException(e, sys)


if __name__ == '__main__':

    di = DataIngestion()
    trainpath, test_path = di.initiate_data_ingestion()

    transformer = DataTransformation()
    trainarr, testarr, _ = transformer.initiate_data_transformation(trainpath, test_path)

    # trainer = ModelTrainer()
    # trainer.initiate_training(trainarr, testarr)

    trainer = EnhanceTrainer()
    trainer.initiate_training(trainarr, testarr)