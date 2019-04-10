#!/usr/bin/python

#python q1_3.py housing_train.txt housing_test.txt

import sys
import numpy as np
import matplotlib.pyplot as plt

# - - - - - Housing Train - - - - -

np.set_printoptions(threshold=np.inf)

def getXY(file, d):
    X = []
    Y = []
    with open(file) as housing_train:
        train_data = np.loadtxt(housing_train, usecols=(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))        

    for row in train_data:
        row = row.tolist()
        for i in range(d):
            row.append(np.random.standard_normal())
            #print(row)
        X.append(row)
    X = np.matrix(X)

    with open(file) as housing_train:
        Y = np.loadtxt(housing_train, usecols=(13))
    
    return X, Y

def getWeightVector(X, Y):
    # w = (XT*X)^-1(XT*y)
    XT = X.T
    XTX = XT * X
    XTX1 = np.linalg.inv(XTX)
    XTX1XT = XTX1 * XT
    w = XTX1XT.dot(Y)
    return w

def getASE(w, X, Y, count):
    #SSE = E(w) = (y-Xw)T(y-Xw)
    Xw = X.dot(w.I)
    yXw = np.subtract(Y, Xw)
    yXwT = yXw.T
    SSE = yXwT.dot(yXw)
    NormSSE = SSE/count
    return NormSSE

def plotGraph(trainASE, testASE, xaxis):
    plt.plot(xaxis,trainASE,'r--',xaxis,testASE,'b--')
    plt.legend(['Training ASE','TestingASE'])
    plt.xlabel('# of random variables')
    plt.ylabel('Average Squared Error')
    plt.title('ASE of Training & Testing when adding d random variables')
    plt.show()

# - - - - - Housing Test - - - - -
'''
X = []
Y = []

with open(sys.argv[2]) as housing_test:
    X = np.loadtxt(housing_test, usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))

with open(sys.argv[2]) as housing_test:
    Y = np.loadtxt(housing_test, usecols=(13))

'''
if __name__ == "__main__":
    dCount = range(2,10)
    trainASE = []
    testASE = []
    for i, d in enumerate(dCount):
        Xtrain, Ytrain = getXY(sys.argv[1], d)
        Xtest, Ytest = getXY(sys.argv[2], d)

        w = getWeightVector(Xtrain, Ytrain)
        print("\nLearned Weight Vector - Training:")
        print(w)

        trainASE.append(getASE(w, Xtrain, Ytrain, 433))
        print("\nAverage Squared Error - Training")
        print(trainASE)
        testASE.append(getASE(w, Xtest, Ytest, 74))
        print("\nAverage Squared Error - Testing")
        print(testASE)

    