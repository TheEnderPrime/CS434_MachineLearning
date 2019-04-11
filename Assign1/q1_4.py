#!/usr/bin/python

#python q1_4.py housing_train.txt housing_test.txt

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

    lines = []
    with open(file) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        data = [int(attr) if i in [3] else float(attr) for i,attr in enumerate(line.split())]
        Y.append([data[-1]])

    return X, np.array(Y)

def getWeightVector(X, Y):
    # w = (XT*X)^-1(XT*y)
    XT = X.T
    XTX = XT.dot(X)
    XTX1 = np.linalg.inv(XTX)
    XTX1XT = XTX1.dot(XT)
    w = XTX1XT.dot(Y)
    return w

def getASE(w, X, Y, count):
	Xw = X * w
	SSE = 0.0
	for i in range(count):
		SSE = pow(Y[i,0] - Xw[i,0], 2)
	return SSE / count

def plotGraph(trainASE, testASE, xaxis):
    plt.plot(xaxis,trainASE,'r--',xaxis,testASE,'b--')
    plt.legend(['Training ASE','TestingASE'])
    plt.xlabel('# of random variables')
    plt.ylabel('Average Squared Error')
    plt.title('ASE of Training & Testing with additional d random variables')
    plt.show()

if __name__ == "__main__":
    dCount = range(2, 11)
    trainASE = []
    testASE = []
    for i, d in enumerate(dCount):
        Xtrain, Ytrain = getXY(sys.argv[1], d)
        Xtest, Ytest = getXY(sys.argv[2], d)
        w = getWeightVector(Xtrain, Ytrain)

        print("\nLearned Weight Vector with - {} - extra variables - Training:".format(d))
        trainASE.append(getASE(w, Xtrain, Ytrain, 433))
        testASE.append(getASE(w, Xtest, Ytest, 74))
    plotGraph(trainASE,testASE,dCount)
    