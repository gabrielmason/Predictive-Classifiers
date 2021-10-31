# Predictive-Classifiers
A set of predictive modeling algorithms using different techniques. Binary Naive Bayes and Decision Trees are complete, and I am current working on a logistic regression
algorithm that will be uploaded once completed

Naive Bayes:
A simple NBC predictive alrogithm. Currently implemented for binary classification, but can easily be extended to support classifying variables 
with more than 2 possible outcomes. Algorithm assumes that all data is discrete. Upon testing the a sample data set, zero-one loss converges to about 0.15 and squared loss converges to about 0.13 with half of the data allocated for training.

Usage: python Naive-Bayes.py \<dataFileName\> \<labelToPredict\>; takes data, splits into training and testing data, runs and analyzes performance of NBC over mutliple trials.
Creates resulting plots
python Naive-Bayes.py \<trainingData\> \<testingData\> \<labelToPredict\>: runs NBC once on training and testing data, prints out zero-one and squared loss

Decision Trees:
A simple NBC predictive alrogithm. Currently implemented for binary classification. Algorithm assumes that all input data is discrete. Upon testing the a sample data set, zero-one loss converges to about 0.12 and squared loss converges to about 0.10 with half of the data allocated for training, however zero-one loss can reach as high as 0.16 depending on the training set randomization.

Usage: python Decision-Tree.py \<dataFileName\> \<labelToPredict\>; takes data, splits into training and testing data, runs and analyzes performance of NBC over mutliple trials. Creates resulting plots
python Decision-Tree.py \<trainingData\> \<testingData\> \<labelToPredict\>: runs NBC once on training and testing data, prints out zero-one and squared loss
