#!/usr/bin/python

#python q1_2.py housing_train.txt

import sys
import csv
import numpy as np

X = []
Y = []

with open(sys.argv[1]) as housing_train:
    
    X = np.loadtxt(housing_train, usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))
    X = np.insert(X, 0, 1, axis=1)

with open(sys.argv[1]) as housing_train:
    Y = np.loadtxt(housing_train, usecols=(13))

    # w = (XT*X)^-1(XT*y)
    XT = X.T
    XTX = XT.dot(X)
    XTX1 = np.power(XTX, -1)
    XTX1XT = XTX1.dot(XT)
    w = XTX1XT.dot(Y)
    print("\nLearned Weight Vector - Training: \n")
    print(w)

    #SSE = E(w) = (y-Xw)T(y-Xw)
    Xw = X.dot(w)
    yXw = np.subtract(Y, Xw)
    yXwT = yXw.T 
    SSE = yXwT.dot(yXw)
    NormSSE = SSE/433
    print("Average Squared Error - Training \n")
    print(NormSSE)


X = []
Y = []

with open(sys.argv[2]) as housing_test:
    
    X = np.loadtxt(housing_test, usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))

with open(sys.argv[2]) as housing_test:
    Y = np.loadtxt(housing_test, usecols=(13))

    # w = (XT*X)^-1(XT*y)
    XT = X.T
    XTX = XT.dot(X)
    XTX1 = np.power(XTX, -1)
    XTX1XT = XTX1.dot(XT)
    w = XTX1XT.dot(Y)
    print("\nLearned Weight Vector - Test:")
    print(w)

    #SSE = E(w) = (y-Xw)T(y-Xw)
    Xw = X.dot(w)
    yXw = np.subtract(Y, Xw)
    yXwT = yXw.T 
    SSE = yXwT.dot(yXw)
    NormSSE = SSE/433
    print("Average Squared Error - Test")
    print(NormSSE)