# -*- coding: utf-8 -*-
from helper import *


# General Remarks
# 1) We are using a 80-training 20-testing split
# 2) Adding more text features to the model by decreasing min_df increases the model accuracy
# 3) Decreasing min_df too much will require significantly more ram (16 GB)
# 4) Link to dataset: https://drive.google.com/open?id=1o3ktyxpWbcGuZHzrpuLhqTlCPZx8vOE-

# Recommended Settings
# 1) best accuracy =>           n = 100,000   min_df = 0.0005
# 1) fast - okay accuracy =>    n = 20,000    min_df = 0.005
# 1) fast results =>            n = 10,000    min_df = 0.01

# only drops 4 columns due to missing channel title name
# title length scaled between [0.00, 1.00]
df = pd.read_csv("dataset.csv").dropna().sample(n=20000, replace=False,
                                                random_state=seed, axis=0)

# selects all features except views which is the response
X = df.loc[:, df.columns != 'views']
# views is the response we will be predicting. It is log10 transformed for faster and better performance
y = np.log10(df['views'])

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed)


# vectorize the text features using TFIDF bag of words and add the new columns to the training data
# prepareTrainData() returns the modified training dataset as well as the transformers to be used for the testing data
print("preparing training:")
prepData = prepareTrainData(x_train)
x_train = prepData[0]

# vectorize the text features using TFIDF bag of words and add the new columns to the testing data
# transform the test data using the transformers which were fitted on the training data.
print("preparing test:")
x_test = prepareTestData(x_test, prepData[1])


# Calling the algorithms. Individual methods and cross validation methods are separated

# doLinearRegression(x_train, y_train, x_test, y_test)
# doSGDRegression(x_train, y_train, x_test, y_test)

# doRidgeRegression(x_train, y_train, x_test, y_test)
# doRidgeCV(x_train, y_train, x_test, y_test)

# doKNNRegression(x_train, y_train, x_test, y_test)
# doKNNGridSearch(x_train, y_train, x_test, y_test)

# doRegressionTree(x_train, y_train, x_test, y_test)
doRegressionTreeGridSearch(x_train, y_train, x_test, y_test)

# doNeuralNetwork(x_train, y_train, x_test, y_test)
# doNeuralCV(x_train, y_train, x_test, y_test)
