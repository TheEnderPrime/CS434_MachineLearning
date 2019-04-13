import sys
import numpy as np
import math
import mpmath
import matplotlib.pyplot as plt

# Run Via This Line:
# python q2_1.py usps-4-9-train.csv usps-4-9-test.csv 1

# Pulls from a file, creates two arrays: X and Y
def getXY(file):
    X = []
    Y = []
    lines = []
    with open(file) as f:
        lines = f.readlines()
    for line in lines:
        data = [int(x) for x in line.strip().split(',')]
        X.append(data[:-1])
        Y.append([data[-1]])

    return np.matrix(X), np.array(Y)

# Plots the accuracy of the train and test models
def plotResults(lambd, trainingAccuracy, testingAccuracy):
    plt.plot(lambd,[t*100 for t in trainingAccuracy],'r--o',lambd,[t*100 for t in testingAccuracy],'b--^')
    plt.xscale('log')
    plt.title('Accuracy of Model over different lambda values')
    plt.xlabel('Lambda Values')
    plt.ylabel('Accuraty (%)')
    plt.legend(['Training Accuracy', 'Testing Accuracy'])
    plt.show()

class LogisticRegression:
    def __init__(self,train,test,lambd):
        (self.trainX, self.trainY) = train
        (self.testX, self.testY) = test
        self.lambd = lambd
        self.learningRate = 0.1
        self.w = np.array([float(1)]*self.trainX.shape[1])
        self.trainingAccuracy = []
        self.testingAccuracy = []

    # built upon the pseudocode on page 17 of the notes
    def train(self):
        X = self.trainX
        Y = self.trainY
        epsilon = 8000
        normGradient = epsilon + 1
        # while until some gradient or a # of iterations
        while ( normGradient >= epsilon ): 
            print("normGradient during training: ", normGradient)
            # set gradient = 0,0,0,0,0,0. . .
            gradient = np.array([0]*X.shape[1])
            # for i in range(X.shape[0]) (for i = 1,...,n)
            for i in range(X.shape[0]): 
                # set Xi to a single data point from the current row
                Xi = np.array(X[i,:])[0]   
                # y^_i or prediction = (1 / (1 + e^-w.T.dot(x_i)))
                prediction = float(1 / (1 + mpmath.exp(-1*np.dot(self.w, Xi))))
                # check for overflow
        # TO DO mpmath can be inaccurate with large numbers so impliment a if else or try catch to do math.exp or mpmath.exp
                # gradient = gradient + (Y^_i - Y_i)*X_i
                gradient = gradient + (prediction - Y[i]) * Xi
            
            # l2-regularization penalty
            l2penalty = []
            for weight in self.w:
                l2penalty.append(math.log(self.lambd * math.pow(weight, 2)))
            np.array(l2penalty)
            
            # weights -= learningRate * gradient 
            self.w -= self.learningRate * ( gradient + l2penalty )
            # normalize gradient to use for while loop
            normGradient = np.linalg.norm(gradient)

    
    # Makes a prediction with the new weights and checks if they are correct 
    def getAccuracy(self): 
        trainedCorrect = 0
        testedCorrect = 0
        
        # this should run through the prediction with the new weights
        # if this prediction equals training or testing Ys then add +1 to the variable

        for i in range(self.trainX.shape[0]):
            Xi = np.array(self.trainX[i,:])[0]
            prediction = float(1 / (1 + mpmath.exp(-1*np.dot(self.w, Xi))))
            if(prediction == self.trainY[i]):
                trainedCorrect += 1
        self.trainingAccuracy.append(trainedCorrect/self.trainY.shape[0])

        for i in range(self.testX.shape[0]):
            Xi = np.array(self.testX[i,:])[0]
            prediction = float(1 / (1 + mpmath.exp(-1*np.dot(self.w, Xi))))
            if(prediction == self.testY[i]):
                testedCorrect += 1
        self.testingAccuracy.append(testedCorrect/self.testY.shape[0])  

        return self.trainingAccuracy, self.testingAccuracy

if __name__ == "__main__":

    # Pull arrays from files
    trainX, trainY = getXY(sys.argv[1])
    testX, testY = getXY(sys.argv[2])
    lambd = []
    lambd.append(float(sys.argv[3]))
    lambd.append(float(sys.argv[4]))
    lambd.append(float(sys.argv[5]))
    lambd.append(float(sys.argv[6]))
    lambd.append(float(sys.argv[7]))

    totalTrainAccuracy = []
    totalTestAccuracy = []

    for l in lambd:
        trainAcc = []
        testAcc = []

        # Create the Logistic Regression model and train it with the batch gradient decent algorithm
        regressionModel = LogisticRegression((trainX, trainY), (testX, testY), l)
        regressionModel.train()

        # Print the weights for analysis
        print("Regression Model Weights:")
        print(regressionModel.w)

        trainAcc, testAcc = regressionModel.getAccuracy()
        totalTrainAccuracy.append(trainAcc)
        totalTestAccuracy.append(testAcc)

    # Plot the accuracy of the model for both training and testing data
    plotResults(lambd, totalTrainAccuracy, totalTestAccuracy)