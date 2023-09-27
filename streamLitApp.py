import streamlit as st
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


def predict_query(form_query):
    data=CustomData(
        Delivery_person_Age = form_query['Delivery_person_Age'],
        Delivery_person_Ratings = form_query['Delivery_person_Ratings'],
        Weather_conditions = form_query['Weather_conditions'],
        Road_traffic_density= form_query['Road_traffic_density'],
        Vehicle_condition = form_query['Vehicle_condition'],
        Type_of_vehicle = form_query['Type_of_vehicle'],
        multiple_deliveries = form_query['multiple_deliveries'],
        Festival = form_query['Festival'],
        City = form_query['City'],
        Pickup_Duration = form_query['Pickup_Duration'],
        Distance = form_query['Distance']
    )

    final_new_data=data.get_data_as_dataframe()
    predict_pipeline=PredictPipeline()
    pred=predict_pipeline.predict(final_new_data)

    result = round(pred[0], 2)

    return result


def home_page(parameters):

    st.title("Delivery Time Prediction App")

    st.write(""" ### Please fill the below information required for prediction! """)

    form_dict = {}
    for param, param_value in parameters.items():
        if param_value == 'text':
            form_dict[param] = st.text_input(param)
        elif hasattr(param_value, '__iter__'):
            form_dict[param] = st.selectbox(param, param_value)


    submit = st.button("Calculate Delivery Time")

    if submit:
        print("Let's Predict.")

        results = predict_query(form_query = form_dict)

        st.subheader(f"The Predicted Time is: {results} minutes.")


if __name__ == '__main__':
    
    parameters = {
            'Delivery_person_Age': 'text',
            'Delivery_person_Ratings': 'text',
            'Vehicle_condition': 'text',
            'multiple_deliveries': 'text',
            'Pickup_Duration': 'text',
            'Distance': 'text',
            'Weather_conditions': ['Fog', 'Stormy', 'Sandstorms', 'Windy', 'Cloudy', 'Sunny'],
            'Road_traffic_density': ['Jam', 'High', 'Medium', 'Low'],
            'Type_of_vehicle': ['motorcycle', 'scooter', 'electric_scooter', 'bicycle'],
            'Festival': ['No', 'Yes'],
            'City': ['Metropolitian', 'Urban', 'Semi-Urban']
            # Add more parameters as needed
        }
    
    home_page(parameters=parameters)