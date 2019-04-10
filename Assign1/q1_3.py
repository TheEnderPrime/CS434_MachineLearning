#!/usr/bin/python

#python q1_3.py housing_train.txt housing_test.txt

import sys
import numpy as np

# - - - - - Housing Train - - - - -


X = []
Y = []

with open(sys.argv[1]) as housing_train:
    
    X = np.loadtxt(housing_train, usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))

with open(sys.argv[1]) as housing_train:
    Y = np.loadtxt(housing_train, usecols=(13))

    # w = (XT*X)^-1(XT*y)
    XT = X.T
    XTX = XT.dot(X)
    XTX1 = np.linalg.inv(XTX)
    XTX1XT = XTX1.dot(XT)
    w = XTX1XT.dot(Y)
    print("\nLearned Weight Vector - Training:")
    print(w)

    #SSE = E(w) = (y-Xw)T(y-Xw)
    Xw = X.dot(w)
    yXw = np.subtract(Y, Xw)
    yXwT = yXw.T 
    SSE = yXwT.dot(yXw)
    NormSSE = SSE/433
    print("\nAverage Squared Error - Training")
    print(NormSSE)


# - - - - - Housing Test - - - - -

X = []
Y = []

with open(sys.argv[2]) as housing_test:
    
    X = np.loadtxt(housing_test, usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))

with open(sys.argv[2]) as housing_test:
    Y = np.loadtxt(housing_test, usecols=(13))

    # w = (XT*X)^-1(XT*y)
    XT = X.T
    XTX = XT.dot(X)
    XTX1 = np.linalg.inv(XTX)
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
    print("\nAverage Squared Error - Test")
    print(NormSSE)