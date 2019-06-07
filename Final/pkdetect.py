import sys
import numpy as np
import math
import matplotlib as mp
import matplotlib.pyplot as plt
from sklearn import tree
import random
mp.use('Agg') 


def decision_tree():

def logistic_regression():

def something_else():


def load_features(file_name, data_length):
    classes, ids = [], []
    f = open(file_name, 'r')
    features = []

    # Skip header line 
    f.readline()

    while True:
        # Read in id and class
        line = f.readline()
        if not line:
            break

        line = line.strip('\n').split('\t')
        features.append([float(i) for i in line[2:]])
        ids.append(line[0])
        classes.append(int(line[1]))

    f.close()

    # Reformat data
    data = np.array(features)
    data = np.append(data, np.array([classes]).T, axis=1)
    data = np.append(data, np.array([ids]).T, axis=1)
    print (data)
    return data

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