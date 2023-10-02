import sys
import os
from src.exeption import CustomException
from src.logger import logger
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)
            logger.info("Preprocessing Done.")

            pred=model.predict(data_scaled)
            logger.info("Prediction Completed.")

            return pred
            

        except Exception as e:
            logger.error(CustomException(e,sys))
            raise e
        
class CustomData:
    def __init__(self,
                 Delivery_person_Age:float,
                 Delivery_person_Ratings:float,
                 Weather_conditions:str,
                 Road_traffic_density:str,
                 Vehicle_condition:int,
                 Type_of_vehicle:str,
                 multiple_deliveries:float,
                 Festival:str,
                 City:str,
                 Pickup_Duration:int,
                 Distance:int):

        # Automatically set instance attributes based on constructor arguments
        
        for arg_name, arg_value in locals().items():
            if arg_name != 'self':
                setattr(self, arg_name, arg_value)

        # print(locals().items())
        # print(self.__dict__.items())


    def get_data_as_dataframe(self):
        try:
            
            df = pd.DataFrame([self.__dict__])
            logger.info(f'Dataframe Gathered\n {df.head().to_string()}')
            # print(self.__dict__.items())
            return df
        except Exception as e:
            logger.error('Exception Occured in Custom Data Gathering.')
            logger.error(CustomException(e,sys))
            raise e


if __name__ == "__main__":
    #logger.info('Prediction Pipeline Started')
    cd = CustomData(Delivery_person_Age = 25,
                 Delivery_person_Ratings = 5.2,
                 Weather_conditions = "Fog",
                 Road_traffic_density = "Jam",
                 Vehicle_condition = "2",
                 Type_of_vehicle = "scooter",
                 multiple_deliveries = 3.0,
                 Festival = "No",
                 City = "Metropolitian",
                 Pickup_Duration = 15,
                 Distance = 10)

    cd_df = cd.get_data_as_dataframe()

    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(cd_df)

    results = round(pred[0], 2)

    print(results)