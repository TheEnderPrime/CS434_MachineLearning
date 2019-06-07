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
mp.use('Agg')

class modelAgg:
	def __init__(self):
		self.models = []
	def add(self, model):
		self.models.append(model)
	def predict(self, file_predict):
		preds[]
		for model in self.models:
			p = model.predict(file_predict)
			preds.append(p)
		probPreds = []
		for instance in range(len(preds[0])):
			prob = sum([preds[i][instance] for i in range(len(preds))] / len(preds)
			probPreds.append(prob)
	return probPreds

def decision_tree(tData, vData):
	print('training on a decision tree...')
	(trainF, TrainL) = tData
	(validateF, validateL) = vData
		
	def buildIt(feat, lables, depth)
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
		validatePred treeData.predict(validateF)
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

    for i in range(5):
        
    


def something_else():


def load_features(filename, validation=0., testing=False):
    dataframe = pandas.read_csv(filename, sep='\t', encoding='utf-8')
    dataframe = dataframe.fillna(0.)
    data = dataframe.values
    if(not testing):
        np.random.shuffle(data)

    if(testing):
        testing_id = data[:, 0]
        testing_features = data[:, 1:]
        testing_features = testing_features.astype('float64')

        return (testing_features, testing_id)
    else:
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
    major_features = load_features('feature103_Train.txt', 103)
    all_features = load_features('featuresall_train.txt', 1053)
    print("All Files Loaded")
    #decision_tree(major_features, 10)


if __name__ == "__main__":
    main()
