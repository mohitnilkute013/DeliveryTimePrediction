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
            logger.error("Exception occured in prediction")
            raise CustomException(e,sys)
        
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
        
        
        self.Delivery_person_Age=Delivery_person_Age
        self.Delivery_person_Ratings=Delivery_person_Ratings
        self.Weather_conditions=Weather_conditions
        self.Road_traffic_density = Road_traffic_density
        self.Vehicle_condition = Vehicle_condition
        self.Type_of_vehicle=Type_of_vehicle
        self.multiple_deliveries=multiple_deliveries
        self.Festival=Festival
        self.City=City
        self.Pickup_Duration=Pickup_Duration
        self.Distance=Distance
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Delivery_person_Age':[self.Delivery_person_Age],
                'Delivery_person_Ratings':[self.Delivery_person_Ratings],
                'Weather_conditions':[self.Weather_conditions],
                'Road_traffic_density':[self.Road_traffic_density],
                'Vehicle_condition':[self.Vehicle_condition],
                'Type_of_vehicle':[self.Type_of_vehicle],
                'multiple_deliveries':[self.multiple_deliveries],
                'Festival':[self.Festival],
                'City':[self.City],
                'Pickup_Duration':[self.Pickup_Duration],
                'Distance':[self.Distance]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logger.info('Dataframe Gathered')
            return df
        except Exception as e:
            logger.error('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)


if __name__ == "__main__":
    #logger.info('Prediction Pipeline Started')
    cd = CustomData(Delivery_person_Age = 25,
                 Delivery_person_Ratings=5.2,
                 Weather_conditions="Fog",
                 Road_traffic_density="Jam",
                 Vehicle_condition="2",
                 Type_of_vehicle="scooter",
                 multiple_deliveries= 3.0,
                 Festival="No",
                 City="Metropolitian",
                 Pickup_Duration=15,
                 Distance=10)

    cd_df = cd.get_data_as_dataframe()
    print(cd_df)

    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(cd_df)

    results = round(pred[0], 2)

    print(results)