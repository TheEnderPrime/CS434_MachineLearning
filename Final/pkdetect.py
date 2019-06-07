import sys
import numpy as np
import math
import matplotlib as mp
import matplotlib.pyplot as plt
import pandas
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import random
mp.use('Agg')


# def decision_tree():


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
        logReg = LogisticRegression(penalty='l2', solver='newton-cg', max_iter=200, fit_intercept=True)
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

    def add(self,model):
        self.models.append(model)

    def predict(self,file_predict):
        preds = []
        for model in self.models:
            p = model.predict(file_predict)
            preds.append(p)
        
        probability_predictions = []
        for instance in range(len(preds[0])):
            probability = sum([preds[i][instance] for i in range(len(preds))]) / len(preds)
            probability_predictions.append(probability)

        return probability_predictions


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
    major_features, major_labels = load_features(
        'data/feature103_Train.txt', validation=0.2)
    #all_features = load_features('featuresall_train.txt', 1053)
    print("All Files Loaded")
    logistic_regression(major_features, major_labels)
    #decision_tree(major_features, 10)


if __name__ == "__main__":
    main()
