# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 08:19:44 2019

@author: danie
"""

from flask import Flask
from flask_restplus import Api, Resource, fields
import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib
import sys
import os
import warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__)
def predict_proba(args):
   train= pd.DataFrame([args], columns=['Year', 'Mileage', 'State', 'Make', 'Model'])
   file='clf.pkl'
   clf = joblib.load(file) 
   le = preprocessing.LabelEncoder()
   for column_name in train.columns:
       if train[column_name].dtype == object:
           train[column_name] = le.fit_transform(train[column_name])
       else:
           pass
   y_pred = pd.DataFrame(clf.predict(train), columns=['Price'])
   return int(y_pred.iloc[0,0])


api = Api(
    app, 
    version='1.0', 
    title='API para la predicción de precios en automoviles',
    description='Autores: Daniel Rojas - Francisco Ortiz')

ns = api.namespace('Características', 
     description='A continuación, introduzca los siguientes valores')
   
parser = api.parser()

parser.add_argument(
    'Year', 
    type=int, 
    required=True, 
    help=' ', 
    location='args')

parser.add_argument(
    'Mileage', 
    type=int, 
    required=True, 
    help=' ', 
    location='args')

parser.add_argument(
    'State', 
    type=int, 
    required=True, 
    help=' ', 
    location='args')

parser.add_argument(
    'Make', 
    type=int, 
    required=True, 
    help=' ', 
    location='args')

parser.add_argument(
    'Model', 
    type=int, 
    required=True, 
    help=' ', 
    location='args')

resource_fields = api.model('Resource', {
    'Precio proyectado': fields.String,
})

@ns.route('/')
class PhishingApi(Resource):
    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "Precio proyectado": predict_proba(args)
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8080)
    