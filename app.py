from flask import Flask, render_template, request
from src.Predictive_Maintenance_RULPrediction.pipelines.prediction_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            unit=float(request.form['unit']),
            time=float(request.form['time']),
            operational_setting_1=float(request.form['op_setting_1']),
            operational_setting_2=float(request.form['op_setting_2']),
            sensor_2=float(request.form['sensor_2']),
            sensor_3=float(request.form['sensor_3']),
            sensor_4=float(request.form['sensor_4']),
            sensor_7=float(request.form['sensor_7']),
            sensor_8=float(request.form['sensor_8']),
            sensor_9=float(request.form['sensor_9']),
            sensor_11=float(request.form['sensor_11']),
            sensor_12=float(request.form['sensor_12']),
            sensor_13=float(request.form['sensor_13']),
            sensor_14=float(request.form['sensor_14']),
            sensor_15=float(request.form['sensor_15']),
            sensor_17=float(request.form['sensor_17']),
            sensor_20=float(request.form['sensor_20']),
            sensor_21=float(request.form['sensor_21'])
        )
        
        df = data.get_data_as_dataframe()
        pipeline = PredictPipeline()
        prediction = pipeline.predict(df)[0]
        
        return render_template('results.html', 
                            unit=data.unit,
                            time=data.time,
                            rul=round(prediction, 2))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)


# use this command to run and see results page
# python app.py & start http://localhost:5001