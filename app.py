from flask import Flask, request, render_template
#from flask_cors import cross_origin
#import jsonify
import sklearn
import requests
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/',methods=['GET'])
#@cross_origin()
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
#@cross_origin()
def predict():
    if request.method == "POST":
        # input variables
        Mohs_Hardness = float(request.form['Mohs_Hardness'])
        Diaphaneity = float(request.form['Diaphaneity'])
        Specific_Gravity = float(request.form['Specific_Gravity'])
        Optical = float(request.form['Optical'])
        Refractive_Index = float(request.form['Refractive_Index'])
        Dispersion = 0
        Molar_Mass = float(request.form['Molar_Mass'])
        Molar_Volume = float(request.form['Molar_Volume'])
        Calculated_Density = Molar_Mass/ Molar_Volume

        #prediction = model.predict([['Mohs_Hardness', 'Diaphaneity', 'Specific_Gravity', 'Optical','Refractive_Index', 'Dispersion', 'Molar_Mass', 'Molar_Volume','Calculated_Density']])
        prediction = model.predict([[Mohs_Hardness, Diaphaneity, Specific_Gravity, Optical, Refractive_Index,Dispersion, Molar_Mass, Molar_Volume,Calculated_Density]])
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text = 'The classified crystal structure {}'.format(output))
    else:

        return render_template('index.html')


if __name__ == "__main__":
   app.run(debug=True)

#if __name__ =="__main__":
 #   app.run(host='0.0.0.0',port=8080)