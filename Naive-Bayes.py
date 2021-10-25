import sys
import urllib.request
import csv
import pandas as pd
import numpy
import urllib.request
import random as rand
import matplotlib.pyplot as plt

#Function takes in three parameters, a training data set file name, a testing data set file name, and the name of
#the target classLabel that we are predicting
def train_and_test_data(train, test, classLabel):
    #Replace all NAN values with "NA"
    train = train.fillna("NA")
    test = test.fillna("NA")
    #Determine default error.
    #We will also use the concatenated list later to ensure that all possible paramters are initialized
    both = pd.concat([train, test])
    overallCounts = both[classLabel].value_counts()
    testCounts = test[classLabel].value_counts()
    mostFreqeunt = 0
    if (overallCounts[0] < overallCounts[1]):
        mostFrequent = 1
    defaultError = 1 - (testCounts[mostFrequent] / test.shape[0])
    attrList = train.columns
    #Learn the NBC with training data set
    #find prior probabilities of Class Label = 0 and Class Label = 1
    train_size = train.shape[0]
    classLabel_counts = train[classLabel].value_counts()
    num0 = classLabel_counts[0]
    num1 = classLabel_counts[1]
    class0_prior = num0/train_size
    class1_prior = num1/train_size

    #Determine parameters
    totalCPD = {}
    for item in attrList:
        if (item == classLabel):
            continue
        #we are looking at each discrete column in the training data
        #determine conditional probability parameters
        CPD = {}
        attrData = train[item]
        #initialize storage for each probability parameter
        #We combined the train and test earlier, use the combined data frame instead of only
        #training to ensure that we have all possible parameter values as to avoid 0 probabilities
        bothAttrData = both[item]
        uniqueAttr = bothAttrData.value_counts(normalize = False)
        for key, val in uniqueAttr.items():
            CPD[key] = [0.00, 0.00]
    
        attrData0 = (train[train[classLabel] == 0])[item]
        attrData1 = (train[train[classLabel] == 1])[item]
        valueCounts0 = attrData0.value_counts(normalize = False)
        valueCounts1 = attrData1.value_counts(normalize = False)

        #Calculate conditional probabilities with laplace smoothing
        for key, val in CPD.items():
            if key in valueCounts0:
                val[0] = (valueCounts0[key] + 1) / (num0 + valueCounts0.size)
            else:
                val[0] = (1) / (num0 + valueCounts0.size)

            if key in valueCounts1:
                val[1] = (valueCounts1[key] + 1) / (num1 + valueCounts1.size)
            else:
                val[1] = (1) / (num1 + valueCounts1.size)

        totalCPD[item] = CPD

    #Test the NBC with test data set
    test_size = test.shape[0]
    numWrong = 0
    squaredLoss = 0
    #calculate probabilities NBC assigns to class label
    #normalize probabilities
    #calculate zero-one and squared loss
    for index, row in test.iterrows():
        true_label = row[classLabel]
        predicted_label = 0
        pclass0 = class0_prior
        pclass1 = class1_prior

        for item in attrList:
            if (item != classLabel):
                attribute = totalCPD[f"{item}"]
                values = attribute[row[item]]
                pclass0 *= values[0]
                pclass1 *= values[1]

        if (pclass0 > pclass1):
            predicted_label = 0
        else:
            predicted_label = 1
    
        if (predicted_label != true_label):
            numWrong += 1

        #print(f"True Label: {true_label}, Predicted Label: {predicted_label}")
        normalized = [pclass0 / (pclass0 + pclass1), pclass1 / (pclass0 + pclass1)]
        squaredLoss += (1 - normalized[true_label]) * (1 - normalized[true_label])

    squaredLoss /= test_size    
    zeroOneLoss = numWrong / test_size
    #print(f"ZERO-ONE LOSS={zeroOneLoss}\nSQUARED-LOSS={squaredLoss}")
    print(f"Test size: {test_size}, numWrong: {numWrong}, Default Error: {defaultError}, ZeroOneLoss: {zeroOneLoss}, Squared loss: {squaredLoss}")
    return [zeroOneLoss, squaredLoss]

#One argument passed in; split data set, analyze NBC preformance
if (len(sys.argv) == 3):
    data = pd.read_csv(sys.argv[1], delimiter=",", header='infer', quotechar="\"")
    size = data.shape[0]
    k = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    zeroOneMeans = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    squaredMeans = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    numTrials = 1
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
    plt.title("NBC Zero One Loss Learning Curve")
    xticks = ["20", "200", "2000", "4000", "6000", "8000","10000"]
    plt.scatter(xticks, zeroOneMeans)
    plt.plot(xticks, zeroOneMeans)
    plt.savefig("/home/mason143/CS373/myProjects/Plots/YelpZeroOneLossNBC.png")
    plt.show()
    plt.clf()
    #squared loss
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Squared Loss")
    plt.title("NBC Squared Loss Learning Curve")
    xticks = ["20", "200", "2000", "4000", "6000", "8000","10000"]
    plt.scatter(xticks, squaredMeans)
    plt.plot(xticks, squaredMeans)
    plt.savefig("/home/mason143/CS373/myProjects/Plots/YelpSquaredLossNBC.png")
    plt.show()
    
#Training set and testing set provided as arguments; run NBC one time
if (len(sys.argv) == 4):
    train = pd.read_csv(sys.argv[1], delimiter=",", header='infer', quotechar="\"")
    test = pd.read_csv(sys.argv[2], delimiter=",", header='infer', quotechar="\"")
    lossScores = train_and_test_data(train, test, sys.argv[3])
