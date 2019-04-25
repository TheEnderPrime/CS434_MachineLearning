import sys
import csv
import math
import numpy as np
from numpy import genfromtxt

TREE_DEPTH = 1
ABSOLUTES = [-1.0, 1.0]

def getBestSplit(data):
    bestIndex, bestValue, bestScore, bestGroups = 100, 100, 100, None

    for i in range(len(data[0]) - 1):
        for row in data:
            groups = split_Data(i, row[i], data)
            gini = giniCalculation(groups)
            if gini < bestScore:
                bestIndex, bestValue, bestScore, bestGroups = i, row[i], gini, groups
    return {"index": bestIndex, "value": bestValue, "groups": bestGroups, "infoGain": bestScore}

def split_Data(index, value, data):
    left, right = list(), list()
    for row in data:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def giniCalculation(sections):
    gini = 0.0
    for value in ABSOLUTES: 
        for section in sections:
            sectionSize = len(section)
            if sectionSize == 0:
                continue
            ratio = [classRow[-1] for classRow in section].count(value) / float(sectionSize)
            gini += (ratio * (1.0 - ratio))
        return gini

#Majority Calculation
def setMajorityClass(section):
	majority = [row[-1] for row in section]
	return max(set(majority), key=majority.count)

def buildTree(node, depthTracker, depthMAX, sizeMIN):
    left, right = node["groups"]
    del(node["groups"])

    if not left or not right:
        node["left"] = node["right"] = setMajorityClass(left + right)
        return
    
    if depthMAX <= depthTracker:
        node["left"], node["right"] = setMajorityClass(left), setMajorityClass(right)
        return

    if len(left) <= sizeMIN:
        node["left"] = setMajorityClass(left)
    else:
        node["left"] = getBestSplit(left)
        buildTree(node["left"], depthTracker + 1, depthMAX, sizeMIN)

    if len(right) <= sizeMIN:
        node["right"] = setMajorityClass(right)
    else:
        node["right"] = getBestSplit(right)
        buildTree(node["right"], depthTracker + 1, depthMAX, sizeMIN)

def leftOrRight(treeBranch, row):
    if row[treeBranch["index"]] < treeBranch["value"]:
        if isinstance(treeBranch["left"], dict):
            return leftOrRight(treeBranch["left"], row)
        else:
            return treeBranch["left"]
    else:
        if isinstance(treeBranch["right"], dict):
            return leftOrRight(treeBranch["right"], row)
        else:
            return treeBranch["right"]

def printTree(treeBranch, depth = 0):
    #treeBranch: The decision tree node
    #depth: Current depth
	if isinstance(treeBranch, dict):
		print ("Depth: %s Branch Index & Value: {X%d = %.4f}" % ((depth*str(depth), (treeBranch["index"] + 1), treeBranch["value"])))
		printTree(treeBranch["left"], depth + 1)
		printTree(treeBranch["right"], depth + 1)
	else:
		print ("Depth: %s, Branch: {%s}" % ((depth * "*", treeBranch)))

#prints out the Depth and gain at each level
def giniGainPrint(treeBranch, depth=0):
	#iterates and prints out entire gini (see source)
	if isinstance(treeBranch, dict):
		print ("Current Depth: " + str(depth))
		print ("Information Gain(Reduction in Entropy) : " + str(treeBranch["infoGain"]))
		giniGainPrint(treeBranch["left"], depth + 1)
		giniGainPrint(treeBranch["right"], depth + 1)

def errorCalculation(tree, data):
    correct = 0
    for row in data:
        guess = leftOrRight(tree, row)
        if guess == row[-1]:
            correct += 1
    return 1 - (float(correct) / len(data))

def print_Results(tree, normalized_testing, normalized_training):
    printTree(tree)
    giniGainPrint(tree)

    print("Train Error: " + str(errorCalculation(tree, normalized_training, )))
    print("Test Error: " + str(errorCalculation(tree, normalized_testing)))
    print("\n")

if __name__ == '__main__':
    # get dataset for training and testing data
    testing_data = genfromtxt('knn_test.csv', delimiter=',')
    training_data = genfromtxt('knn_train.csv', delimiter=',')
    try:
        TREE_DEPTH = int(sys.argv[3])
    except: 
        TREE_DEPTH = 1
    
    if TREE_DEPTH > 6:
        TREE_DEPTH = 6

    # deliniate data into x and y axis for both data sets
    x_training = np.array(training_data[:,1:31])
    y_training = np.array(training_data[:,0:1])
    x_testing = np.array(testing_data[:,1:31])
    y_testing = np.array(testing_data[:,0:1])

    x_training /= x_training.sum(axis=1)[:,np.newaxis]
    x_testing /= x_testing.sum(axis=1)[:,np.newaxis]
    normalized_training = np.append(x_training,y_training,axis=1)
    normalized_testing = np.append(x_testing,y_training,axis=1)

    # Decision Stump
    tree = getBestSplit(normalized_training)
    buildTree(tree, 1, 1, 10)
    print ("Data for Stump:")
    print_Results(tree, normalized_testing, normalized_training)

    # Decision Tree with 
    tree = getBestSplit(normalized_training)
    buildTree(tree, 1, TREE_DEPTH, 10)
    print ("Data for Tree: Depth up to " + str(TREE_DEPTH))
    print_Results(tree, normalized_testing, normalized_training)