# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 21:02:40 2014

@author: HaydenAdmin
"""

import csv
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn import linear_model

def get_X(row):
    numeric_feature = ['Tenure','Age']
    X ={}
    for key,value in row.items():
        if key not in numeric_feature:
            X[key] = value
    return X        



if __name__ == '__main__':
#    filename = 'DS_MiniProject_ANON.csv'
##    D = 2**20
#    with open(filename) as f:
#        reader = csv.DictReader(f)
#        X = []
#        Y = []
#        for row in reader:
#            Y.append (int(row['Call_Flag'] ))
#            del row['DATE_FOR']
##            X.append(row.items())
#            X.append(get_X(row))
##                print key 
##                index = int(value, 8) % D
##                X.append(index)
    v = DictVectorizer(sparse=False)
    
    n_train = int(len(X) * .6)
    
    X_train = v.fit_transform(X[:n_train] )
    X_test = v.transform(X[ n_train:])
    Y_train = Y[:n_train]
    Y_test = Y[ n_train:]
#    clf = linear_model.SGDClassifier(loss = 'log', shuffle = True )
    clf = linear_model.LogisticRegression()
    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    print sum(Y_pred!=Y_test)