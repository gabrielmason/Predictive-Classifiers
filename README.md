# Predictive-Classifiers
A set of predictive modeling algorithms using various techniques. Binary Naive Bayes and Decision Trees are complete, and I am currently working on a logistic regression algorithm that will be uploaded once completed

**Naive Bayes:**

A simple NBC predictive alrogithm. Currently implemented for binary classification, but can easily be extended to support classifying variables 
with more than 2 possible outcomes. Algorithm assumes that all data is discrete. Upon testing a sample data set, zero-one loss converges to about 0.15 and squared loss converges to about 0.13 with half of the data allocated for training.

![Screen Shot 2021-10-31 at 8 30 26 PM](https://user-images.githubusercontent.com/54636576/139606825-45923860-224f-4d42-9456-daf4cadbd216.png)
![Screen Shot 2021-10-31 at 8 30 34 PM](https://user-images.githubusercontent.com/54636576/139606822-c1c1bfea-ebd4-4e09-8f5b-ef8bf779aa72.png)

Usage: python Naive-Bayes.py \<dataFileName\> \<labelToPredict\>; takes data, splits into training and testing data, runs and analyzes performance of NBC over mutliple trials with randomized data splits for each trial; Creates resulting plots

python Naive-Bayes.py \<trainingData\> \<testingData\> \<labelToPredict\>: runs NBC once on training and testing data, prints out zero-one and squared loss


**Decision Trees:**

A decision tree modeling alrogithm. Currently implemented for binary classification. Algorithm assumes that all input data is discrete. Upon testing a sample data set, zero-one loss converges to about 0.12 and squared loss converges to about 0.11 with half of the data allocated for training

![Screen Shot 2021-10-31 at 8 29 40 PM](https://user-images.githubusercontent.com/54636576/139606772-ebea2b7a-aaab-4515-befb-f47d028ce9de.png)
![Screen Shot 2021-10-31 at 8 28 53 PM](https://user-images.githubusercontent.com/54636576/139606773-94c0e50c-f00f-4a9b-97db-69ec44c92698.png)

Usage: python Decision-Tree.py \<dataFileName\> \<labelToPredict\>; takes data, splits into training and testing data, runs and analyzes performance of decision tree over mutliple trials with randomized data splits for each trial; Creates resulting plots

python Decision-Tree.py \<trainingData\> \<testingData\> \<labelToPredict\>: runs decision tree once on training and testing data, prints out zero-one and squared loss


