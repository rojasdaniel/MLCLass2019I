# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 08:19:34 2019

@author: danie
"""

import pandas as pd
from sklearn.externals import joblib
import sys
import os

def predict_proba(args):
    train=pd.DataFrame.from_dict(args)
#    clf = joblib.load(os.path.dirname(__file__) + '/phishing_clf.pkl') 
#    for column_name in train.columns:
#        if train[column_name].dtype == object:
#            train[column_name] = le.fit_transform(train[column_name])
#        else:
#            pass
#    p1 = clf.predict_proba(train)
#    y_pred = pd.DataFrame(p1, columns=['Price'])
    y_pred-train
    return y_pred


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add an URL')
        
    else:

        url = sys.argv[1]

        p1 = predict_proba(url)
        
        print(url)
        print('Probability of Phishing: ', p1)