# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:25:30 2019

@author: danie
"""
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from sklearn import preprocessing

%matplotlib inline

train = pd.read_csv('https://github.com/albahnsen/PracticalMachineLearningClass/raw/master/datasets/dataTrain_carListings.zip')
test = pd.read_csv('https://github.com/albahnsen/PracticalMachineLearningClass/raw/master/datasets/dataTest_carListings.zip', index_col=0)

#we use log function which is in numpy

train['Price'] = np.log1p(train['Price'])

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


plt.subplots(figsize=(12,9))
sns.distplot(train['Price'], fit=stats.norm)

plt.subplots(figsize=(12,9))
sns.distplot(train['Price'], fit=stats.norm)
# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(train['Price'])
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
fig = plt.figure()
stats.probplot(train['Price'], plot=plt)
plt.show()

#Let's check if the data set has any missing values. 

train.columns[train.isnull().any()]

#plot of missing value attributes
plt.figure(figsize=(12, 6))
sns.heatmap(train.isnull())
plt.show()

train_corr = train.select_dtypes(include=[np.number])
train_corr.shape
#Coralation plot
corr = train_corr.corr()
plt.subplots(figsize=(20,9))
sns.heatmap(corr, annot=True)

print("Find most important features relative to target")
corr = train.corr()
corr.sort_values(['Price'], ascending=False, inplace=True)
corr.Price


feature_cols = ['Year', 'Mileage', 'State', 'Make', 'Model']
X = train[feature_cols]
y = train.Price

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,  test_size=0.3)

#Linear Regression
from sklearn import linear_model
model1 = linear_model.LinearRegression()
model1.fit(X_train, y_train)
print("Predict value " + str(model1.predict([X_test.iloc[12]])))
print("Real value " + str(y_test.iloc[12]))
print("Accuracy --> ", model1.score(X_test, y_test)*100)


# Random Forest
from sklearn.ensemble import RandomForestRegressor
model2 = RandomForestRegressor(n_estimators=200)
model2.fit(X_train, y_train)
y_pred = pd.DataFrame(model2.predict(test), index=test.index, columns=['Price'])

print("Predict value " + str(model2.predict([X_test.iloc[12]])))
print("Real value " + str(y_test.iloc[12]))
print("Accuracy --> ", model2.score(X_test, y_test)*100)

#GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)
GBR.fit(X_train, y_train)
print("Accuracy --> ", GBR.score(X_test, y_test)*100)
