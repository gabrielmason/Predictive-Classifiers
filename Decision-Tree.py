import sys
import urllib.request
import csv
import pandas as pd
import numpy
import urllib.request
import random as rand
import matplotlib.pyplot as plt

maxHeight = 4

class tree:
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1
        self.bestSplit = "none"
        self.children = {}

    def set_best_split(self, bestSplit):
        self.bestSplit = bestSplit

    def set_children(self, children):
        self.children = children
    
    def print(self):
        print(f"p0: {self.p0,}, p1: {self.p1}, bestSplit: {self.bestSplit}")
        for key, val in self.children.items():
            val.print()


def gini(x, y):
    return 1 - ((x/(x+y)) * (x/(x+y)) + (y/(x+y)) * (y/(x+y)))


def build_tree(train, classLabel, attrList, height):
    train_size = train.shape[0]
    numCols = len(attrList)
    valueCounts = train[classLabel].value_counts()
    if (0 not in valueCounts):
        valueCounts[0] = 0
    if (1 not in valueCounts):
        valueCounts[1] = 0

    currentGini = gini(valueCounts[0], valueCounts[1])

    p0 = valueCounts[0]/(valueCounts[0]+valueCounts[1])
    p1 = valueCounts[1]/(valueCounts[0]+valueCounts[1])

    currentTree = tree(p0, p1)
    if (height > maxHeight): #stop building tree at maxHeight
        return currentTree

    maxGain = 0
    bestSplit = "none"

    for attr in attrList: #determine best split
        if (attr == classLabel):
            continue

        gain = 0
        attrData = train[attr]
        uniqueValues = attrData.unique()
        valueOccurrences = attrData.value_counts()
        
        for value in uniqueValues: #iterate through each possible parameter value, making a new tree for each; find gain with gini
            split = train.loc[train[attr] == value]
            splitCounts = split[classLabel].value_counts()
            if (0 not in splitCounts):
                splitCounts[0] = 0
            if (1 not in splitCounts):
                splitCounts[1] = 0

            Sa = valueOccurrences[value]
            gain += (Sa/train_size) * gini(splitCounts[0], splitCounts[1])

        gain = currentGini - gain
        if (gain > maxGain):
            maxGain = gain
            bestSplit = attr

    #build trees of children according to best split
    if (maxGain < 0.05): #stop building tree if gain isn't high enough
        return currentTree
    if (bestSplit == "none"):
        return currentTree

    currentTree.set_best_split(bestSplit)
    children = {}
    uniqueValues = train[bestSplit].unique()
    temp = attrList.copy()
    attrList.remove(bestSplit)

    for value in uniqueValues:
        if (value == classLabel):
            continue
        split = train.loc[train[bestSplit] == value]
        children[value] = build_tree(split.drop(bestSplit, 1, inplace = False), classLabel, attrList, height + 1)
    
    currentTree.set_children(children)      
    return currentTree
             

def train_and_test_data(train, test, classLabel):
    #Train Decision Tree Model
    #Replace all NAN values with "NA"
    train = train.fillna("NA")
    test = test.fillna("NA")
    attrList = list(train.columns)
    train_size = train.shape[0]
    numCols = len(attrList)
    #Determine default error.
    both = pd.concat([train, test])
    overallCounts = both[classLabel].value_counts()
    testCounts = test[classLabel].value_counts()
    mostFreqeunt = 0
    if (overallCounts[0] < overallCounts[1]):
        mostFrequent = 1
    defaultError = 1 - (testCounts[mostFrequent] / test.shape[0])
    decisionTree = build_tree(train, classLabel, attrList, 0)
    #decisionTree.print()

    #Test DecisionTree Model
    test_size = test.shape[0]
    attrList = list(test.columns)
    numWrong = 0
    squaredLoss = 0
    numIgnored = 0
    for index, row in test.iterrows():
        true_label = row[classLabel]
        predicted_label = 0

        current = decisionTree
        for i in range(0, maxHeight):
            if (current.bestSplit == "none"):
                break
            if (row[current.bestSplit] not in current.children):
                numIgnored += 1
                break
            current = current.children[row[current.bestSplit]]

        pclass0 = current.p0
        pclass1 = current.p1

        if (pclass0 > pclass1):
            predicted_label = 0
        else:
            predicted_label = 1
    
        if (predicted_label != true_label):
            numWrong += 1

        normalized = [pclass0 / (pclass0 + pclass1), pclass1 / (pclass0 + pclass1)]
        squaredLoss += (1 - normalized[true_label]) * (1 - normalized[true_label])

    squaredLoss /= test_size    
    zeroOneLoss = numWrong / test_size
    #print(f"ZERO-ONE LOSS={zeroOneLoss}\nSQUARED-LOSS={squaredLoss}")
    print(f"Test size: {test_size}, Number ignored: {numIgnored}, default error: {defaultError}, tree height: {maxHeight} numWrong: {numWrong}, ZeroOneLoss: {zeroOneLoss}, Squared loss: {squaredLoss}")
    return [zeroOneLoss, squaredLoss]
    


#One argument passed in; split data set, analyze NBC preformance
if (len(sys.argv) == 3):
    data = pd.read_csv(sys.argv[1], delimiter=",", header='infer', quotechar="\"")
    size = data.shape[0]
    k = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    zeroOneMeans = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    squaredMeans = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    numTrials = 5
    for i in range (len(k)):
        print(k[i])
        for j in range (numTrials):
            temp = data.copy(deep=True)
            train = temp.sample(frac = k[i])
            test = temp.drop(train.index)
            train.to_csv("Data/YelpTrain.csv", index=False)
            test.to_csv("Data/YelpTest.csv", index=False)
            lossScores = train_and_test_data(train, test, sys.argv[2])
            zeroOneMeans[i] += lossScores[0]
            squaredMeans[i] += lossScores[1]
        zeroOneMeans[i] /= numTrials
        squaredMeans[i] /= numTrials
        
    #plot learning curves
    #zero one loss
    fig, ax = plt.subplots()
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Zero One Loss")
    plt.title("Decision Tree Zero One Loss Learning Curve")
    xticks = ["20", "200", "2000", "4000", "6000", "8000","10000"]
    plt.scatter(xticks, zeroOneMeans)
    plt.plot(xticks, zeroOneMeans)
    plt.show()
    plt.clf()
    #squared loss
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Squared Loss")
    plt.title("Decision Tree Squared Loss Learning Curve")
    xticks = ["20", "200", "2000", "4000", "6000", "8000","10000"]
    plt.scatter(xticks, squaredMeans)
    plt.plot(xticks, squaredMeans)
    plt.show()
    
#Training set and testing set provided as arguments; run NBC one time
if (len(sys.argv) == 4):
    train = pd.read_csv(sys.argv[1], delimiter=",", header='infer', quotechar="\"")
    test = pd.read_csv(sys.argv[2], delimiter=",", header='infer', quotechar="\"")
    train_and_test_data(train, test, sys.argv[3])
