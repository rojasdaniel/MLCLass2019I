# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 19:10:13 2019

@author: danie
"""

train = pd.read_csv('https://github.com/albahnsen/PracticalMachineLearningClass/raw/master/datasets/dataTrain_carListings.zip')
test = pd.read_csv('https://github.com/albahnsen/PracticalMachineLearningClass/raw/master/datasets/dataTest_carListings.zip', index_col=0)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
accuracy_scores = []
feature_range = [0.5,0.05,0.005]
estimator_range = range(10, 310, 10)
max_depth_range = range(1, 21)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_name in train.columns:
    if train[column_name].dtype == object:
        train[column_name] = le.fit_transform(train[column_name])
    else:
        pass
le = preprocessing.LabelEncoder()
for column_name in test.columns:
    if train[column_name].dtype == object:
        train[column_name] = le.fit_transform(train[column_name])
    else:
        pass
#train[cat] = train[cat].apply(lambda x: x.cat.codes)

fig, ax = plt.subplots()
ax.scatter(x = train['Mileage'], y = train['Price'])
plt.show()

train = train.drop(train[(train['Mileage']>1000000)].index)

feature_cols = ['Year', 'Mileage', 'State', 'Make', 'Model']
X = train[feature_cols]

y = train.Price

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,  test_size=0.3)
for depth in max_depth_range:
    for estimator in estimator_range:
        for feature in feature_range:
            clf = GradientBoostingRegressor(n_estimators=estimator, max_features='sqrt', max_depth=depth, learning_rate= feature, random_state=1)
            clf.fit(X_train, y_train)
            cars_predictions = clf.predict(X_test)
            accuracy_scores.append([depth, estimator, feature, np.sqrt(mean_squared_error(y_test, cars_predictions))])
            print([depth, estimator, feature, np.sqrt(mean_squared_error(y_test, cars_predictions))])

accuracy_scores = pd.DataFrame(accuracy_scores, columns=['Depth', 'Estimator','Feature', 'RMSE'])
print('Best Accuracy Score: ')
print(accuracy_scores.loc[accuracy_scores['RMSE'].idxmin()])