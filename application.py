import pickle
from flask import Flask, request, jsonify, render_template

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 

application = Flask(__name__)
app = application

## Import ridge regressor model and standard Scaler pickle file

ridge_model = pickle.load(open('models/ridge.pkl' , 'rb'))
standard_Scaler = pickle.load(open('models/scaler.pkl' , 'rb'))


## Home page Route
@app.route('/')
def index():
    return render_template('index.html')

## Forms page
@app.route('/predictdata' , methods = ['GET','POST'])
def predict_datapoints():
    if request.method == 'POST':
        Temprature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = standard_Scaler.transform([[Temprature , RH , Ws , Rain , FFMC , DMC , ISI , Classes , Region]])
        result1 = ridge_model.predict(new_data_scaled)

        return render_template('home.html', result = result1[0])
    
    else:
        return render_template('home.html')
    
if __name__ == "__main__":
    app.run(host = "0.0.0.0")

    




