# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:16:52 2019

@author: danie
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import os
os.chdir('C:/Users/danie/OneDrive - Universidad de Los Andes/vanaliticadev/SEMESTRE 2/MLClass/P1')

train = pd.read_csv('https://github.com/albahnsen/PracticalMachineLearningClass/raw/master/datasets/dataTrain_carListings.zip')
test = pd.read_csv('https://github.com/albahnsen/PracticalMachineLearningClass/raw/master/datasets/dataTest_carListings.zip', index_col=0)

from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')

le = preprocessing.LabelEncoder()
for column_name in train.columns:
    if train[column_name].dtype == object:
        train[column_name] = le.fit_transform(train[column_name])
    else:
        pass
for column_name in test.columns:
    if test[column_name].dtype == object:
        test[column_name] = le.fit_transform(test[column_name])
    else:
        pass
feature_cols = ['Year', 'Mileage', 'State', 'Make', 'Model']
X = train[feature_cols]
y = train.Price

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,  test_size=0.3)
    
GBoost = GradientBoostingRegressor(n_estimators=20, learning_rate=1,
                                   max_depth=4, max_features='sqrt', random_state =5)
GBoost.fit(X_train, y_train)
y_pred = pd.DataFrame(GBoost.predict(test), index=test.index, columns=['Price'])
#y_pred.to_csv('test_submission.csv', index_label='ID')


from sklearn.externals import joblib

joblib.dump(GBoost, 'model_deployment/clf.pkl', compress=3)
