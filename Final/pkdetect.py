import sys
import numpy as np
import math
import matplotlib as mp
import matplotlib.pyplot as plt
import pandas
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as decTreeClass
import random
from joblib import dump, load
mp.use('Agg')

class modelAgg:
	def __init__(self):
		self.models = []
	def add(self, model):
		self.models.append(model)
	def predict(self, file_predict):
		preds = []
		for model in self.models:
			p = model.predict(file_predict)
			preds.append(p)
		probPreds = []
		for instance in range(len(preds[0])):
			prob = sum([preds[i][instance] for i in range(len(preds))]) / len(preds)
			probPreds.append(prob)
		
		return probPreds

def decision_tree(tData, vData):
	print('training on a decision tree...')
	(trainF, TrainL) = tData
	(validateF, validateL) = vData
		
	def buildIt(feat, lables, depth):
		clf_ent = DecisionTreeClassifier(criterion="entropy",max_depth=d)
		clf_ent.fit(features,labels)
		return clf_ent
		
	minD, maxD, = 1,12 #experiment with depths?
	print("Trying trees from depth {} to {}".format(minD,maxD))
		
	treeDataData = modelAgg()
	for depth in range(minD, maxD+1):
		treeData = buildIt(trainF, trainL, depth)	
		trainPred = treeData.predict(trainF)
		trainAcc = sum([1 if(trainL[i]==p) else 0 for i,p in enumerate(trainPred)])/len(trainL)
		validatePred = treeData.predict(validateF)
		validateAcc =sum([1 if(validateL[i]==p) else 0 for i,p in enumerate(validatePred)])/len(validateL)
		print('Decision Tree: Accuracy with d={}: (t:{:.4f}), (v:{:.4f})'.format(d, trainAcc, validateAcc))
		treeDataData.add(treeData)
	
	return treeDataData
		


def logistic_regression(training, validation):
    print('Logistic Regression')

    (trainingFeatures, trainingLabels) = training
    (validationFeatures, validationLabels) = validation

    print('-normalizing features-')

    scalar = StandardScaler()

    scalar.fit(trainingFeatures)
    trainingFeatures = scalar.transform(trainingFeatures)
    scalar.fit(validationFeatures)
    validationFeatures = scalar.transform(validationFeatures)

    print('-running algorithm-')

    models = ModelClass()
    for i in range(5):
        logReg = LogisticRegression(
            penalty='l2', solver='newton-cg', max_iter=200, fit_intercept=True)
        logReg.fit(trainingFeatures, trainingLabels)

        trainingPredictions = logReg.predict(trainingFeatures)
        trainingAccuracy = sum([1 if(trainingLabels[i] == p) else 0 for i, p in enumerate(
            trainingPredictions)])/len(trainingLabels)

        validationPredictions = logReg.predict(validationFeatures)
        validationAccuracy = sum([1 if(validationLabels[i] == p) else 0 for i, p in enumerate(
            validationPredictions)])/len(validationLabels)

        print('Logistic Regression (Attempt {}): Accuracy: (t:{:.4f}), (v:{:.4f})'.format(
            i, trainingAccuracy, validationAccuracy))

        models.add(logReg)
    print('-logistic regression finished-')
    return models


# def something_else():


class ModelClass:
    def __init__(self):
        self.models = []

    def add(self, model):
        self.models.append(model)

    def predict(self, file_predict):
        preds = []
        for model in self.models:
            p = model.predict(file_predict)
            preds.append(p)

        probability_predictions = []
        for instance in range(len(preds[0])):
            probability = sum([preds[i][instance]for i in range(len(preds))]) / len(preds)
            probability_predictions.append(probability)

        return probability_predictions


def load_features_for_testing(testingData):
    dataframe = pandas.read_csv(testingData, sep='\t', encoding='utf-8')
    dataframe = dataframe.fillna(0.)
    data = dataframe.values

    testingId = data[:, 0]
    testingFeatures = data[:, 1:]
    testingFeatures = testingFeatures.astype('float64')

    return (testingFeatures, testingId)


def load_features(filename, validation=0., testing=False):
    dataframe = pandas.read_csv(filename, sep='\t', encoding='utf-8')
    dataframe = dataframe.fillna(0.)
    data = dataframe.values
    
    np.random.shuffle(data)

    num_for_validation = int(validation * dataframe.shape[0])
    training_features = data[num_for_validation:, 2:]
    validation_features = data[:num_for_validation, 2:]
    training_labels = data[num_for_validation:, 1]
    validation_labels = data[:num_for_validation, 1]
    training_features = training_features.astype('float64')
    validation_features = validation_features.astype('float64')
    training_labels = training_labels.astype('int')
    validation_labels = validation_labels.astype('int')

    if(validation > 0):
        return (training_features, training_labels), (validation_features, validation_labels)
    else:
        return (training_features, training_labels)


def main():
    # Load in data
    random.seed()
    print("Files Loading")
    major_features, major_labels = load_features('data/feature103_Train.txt', validation=0.2)
    #all_features = load_features('featuresall_train.txt', 1053)
    print("All Files Loaded")
    LogRegModel = logistic_regression(major_features, major_labels)
	
	
    #decision_tree(major_features, 10)

    print("Dumping Models To File")
    dump(LogRegModel, "logreg_model")

    print("Testing and Predictions For Logrithmic Regression")
    logRegTesting = load_features_for_testing('data/feature103_Train.txt')
    
    (features,ids) = logRegTesting
    model = load('logreg_model')
    
    scalar = StandardScaler()
    scalar.fit(features)
    features = scalar.transform(features)

    predictions = model.predict(features)
    with open('logreg_predictions_103','w+') as f:
        for i,prediction in enumerate(predictions):
            f.write('{},{}\n'.format(ids[i],prediction))
			
	

if __name__ == "__main__":
    main()
