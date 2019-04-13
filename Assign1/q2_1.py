import sys
import numpy as np
import math
import mpmath
import matplotlib.pyplot as plt

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


def plotResults(trainingAccuracy, testingAccuracy):
    plt.plot()
    plt.title()
    plt.xlabel()
    plt.ylabel()
    plot.legend()
    plt.show()

#def plotGraph(trainASE, testASE, xaxis):
#    plt.plot(xaxis,trainASE,'r--',xaxis,testASE,'b--')
#    plt.legend(['Training ASE','TestingASE'])
#    plt.xlabel('# of random variables')
#   plt.ylabel('Average Squared Error')
#    plt.title('ASE of Training & Testing with additional d random variables')
#    plt.show()    

class LogisticRegression:
    def __init__(self,train,test,learnRate):
        (self.trainX, self.trainY) = train
        (self.testX, self.testY) = test
        self.learningRate = learnRate
        self.w = np.array([float(0)]*self.trainX.shape[1])
        self.trainingAccuracy = []
        self.testingAccuracy = []

    def train(self):
        X = self.trainX
        Y = self.trainY
        epsilon = 3000
        normGradient = epsilon + 1
        # while until some gradient or a # of iterations
        while ( normGradient >= epsilon ): 
            print("normGradient during training: ", normGradient)
            #   set gradient = 0,0,0,0,0,0. . .
            gradient = np.array([0]*X.shape[1])
        #   for i in range(X.shape[0]) (for i = 1,...,n)
            for i in range(X.shape[0]): 
                #   set Xi to a single data point from the current row
                Xi = np.array(X[i,:])[0]   
        #       y^_i or prediction = (1 / (1 + e^-w.T.dot(x_i)))
                prediction = float(1 / (1 + mpmath.exp(-1*np.dot(self.w, Xi))))
        #       check for overflow
# TO DO mpmath can be inaccurate with large numbers so impliment a if else or try catch to do math.exp or mpmath.exp
        #       gradient = gradient + (Y^_i - Y_i)*X_i
                gradient = gradient + (prediction - Y[i]) * Xi
        #   weights -= learningRate * gradient 
            self.w -= learningRate * gradient
        #   normalize gradient to use for while loop
            normGradient = np.linalg.norm(gradient)

            self.getAccuracy()
    
    def getAccuracy(self): #gets accuracy for plotting later 
        trainedCorrect = 0
        testedCorrect = 0

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
                trainedCorrect += 1
        # this should run through the prediction with the new weights
        # if this prediction equals training or testing Ys then added ++ to a variable
        self.testingAccuracy.append(testedCorrect/self.testY.shape[0])  
        print("gettingAccuracy:")

if __name__ == "__main__":
    trainX, trainY = getXY(sys.argv[1])
    testX, testY = getXY(sys.argv[2])
    learningRate = float(sys.argv[3])

    regressionModel = LogisticRegression((trainX, trainY), (testX, testY), learningRate)
    regressionModel.train()

    print(regressionModel.w)

    #plotResults(model.trainingAccuracy, model.testingAccuracy)