import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.exeption import CustomException
from src.logger import logger
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logger.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['Weather_conditions', 'Road_traffic_density', 'Type_of_vehicle', 'Festival', 'City']
            numerical_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition',
            'multiple_deliveries', 'Pickup_Duration', 'Distance']
            
            # Define the custom ranking for each ordinal variable
            Weather_conditions_cat = ['Fog', 'Stormy', 'Sandstorms', 'Windy', 'Cloudy', 'Sunny']
            Weather_conditions_cat.reverse()
            Road_traffic_density_cat = ['Jam', 'High', 'Medium', 'Low']
            Road_traffic_density_cat.reverse()
            Type_of_order_cat = ['Drinks', 'Snack', 'Meal', 'Buffet']
            Type_of_vehicle_cat = ['motorcycle', 'scooter', 'electric_scooter', 'bicycle']
            Festival_cat = ['No', 'Yes']
            City_cat = ['Metropolitian', 'Urban', 'Semi-Urban']
            City_cat.reverse()
            
            logger.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[Weather_conditions_cat,Road_traffic_density_cat, Type_of_vehicle_cat, Festival_cat, City_cat])),
                ('scaler',StandardScaler())  #Type_of_order_cat
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            logger.info('Pipeline Completed')

            return preprocessor

        except Exception as e:
            logger.error("Error in Preparing Data Transformation Object")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info('Read train and test data completed')
            logger.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logger.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logger.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            # Target Column to be predicted
            target_column_name = 'Time_taken (min)'
            drop_columns = [target_column_name, 'Restaurant_latitude',
            'Restaurant_longitude', 'Delivery_location_latitude',
            'Delivery_location_longitude', 'Day', 'Month', 'Order_Hour', 'Order_Min',
            'Pick_Hour', 'Pick_Min', 'Type_of_order']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logger.info("Applying preprocessing object on training and testing datasets.")
            
            #concatenates side by side
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logger.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logger.error("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)