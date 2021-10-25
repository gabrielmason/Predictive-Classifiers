# Predictive-Classifiers
A set of predictive modeling algorithms using different techniques. Binary Naive Bayes is complete, and I am current working on a decision tree
algorithm that will be uploaded once completed

Naive Bayes:
A simple NBC predictive alrogithm. Currently implemented for binary classification, but can easily be extended to support classifying variables 
with more than 2 possible outcomes. Algorithm assumes that all data is discrete. Upon testing the a sample data set, squared loss converges to about 0.13 with half of the data allocated for training.

Usage: python NaiveBayes.py <data file name> <class Label to predict>; takes data, splits into training and testing data, runs and analyzes performence of NBC over mutliple trials.
python NaiveBayes.py <training data> <testing data> <class label to predict>: runs NBC once on training and testing data, prints out zero-one and squared loss
