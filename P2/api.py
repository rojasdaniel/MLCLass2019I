# coding=utf-8
from flask import Flask
from flask_restplus import Api, Resource, fields
import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib
import sys
import os
import warnings
from P2MovieGenrePrediction import *
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__)
def predict_probar(args):
   train= pd.DataFrame([args], columns=['title', 'plot'])
   a = predict_probar2(train)
   return a.to_dict('list')


api = Api(
    app, 
    version='1.0', 
    title='API para la predicción de géneros en películas',
    description='Autores: Daniel Rojas - Francisco Ortiz')

ns = api.namespace('Características', 
     description='A continuación, introduzca los siguientes valores')
   
parser = api.parser()

parser.add_argument(
    'title', 
    type=str, 
    required=True, 
    help=' ', 
    location='args')

parser.add_argument(
    'plot', 
    type=str, 
    required=True, 
    help=' ', 
    location='args')


resource_fields = api.model('Resource', {
    'Categoria proyectada': fields.String,
})

@ns.route('/')
class PhishingApi(Resource):
    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "Categoria proyectada": predict_probar(args)
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
    