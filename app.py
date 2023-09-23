from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application


@app.route('/', methods=['GET', 'POST'])
def home_page():

    title = 'Delivery Time Prediction App'

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

    if request.method=='GET':
        return render_template('index.html', title=title, parameters=parameters)
    else:
        data=CustomData(
            Delivery_person_Age=float(request.form.get('Delivery_person_Age')),
            Delivery_person_Ratings = float(request.form.get('Delivery_person_Ratings')),
            Weather_conditions = request.form.get('Weather_conditions'),
            Road_traffic_density= request.form.get('Road_traffic_density'),
            Vehicle_condition = int(request.form.get('Vehicle_condition')),
            Type_of_vehicle = request.form.get('Type_of_vehicle'),
            multiple_deliveries = float(request.form.get('multiple_deliveries')),
            Festival = request.form.get('Festival'),
            City = request.form.get('City'),
            Pickup_Duration = int(request.form.get('Pickup_Duration')),
            Distance = int(request.form.get('Distance'))
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results = f"The predicted result is: {round(pred[0], 2)}"

        return render_template('index.html', title=title, parameters=parameters, final_result=results)


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True, port=5000)

