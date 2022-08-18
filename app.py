# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 11:59:01 2022

@author: Deblina
"""
from sklearn.model_selection import train_test_split
from flask import Flask,render_template,url_for,request
import pandas as pd 

import pickle

# load the model from disk
loaded_model=pickle.load(open('random_forest_regression_model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('pro.html')

@app.route('/predict')
def predict():
    df=pd.read_csv('Downloads/cities (1).csv')
    my_prediction=loaded_model.predict(df.iloc[:,:-1].values)
    my_prediction=my_prediction.tolist()
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)