# -*- coding: utf-8 -*-
from helper import *

# df = pd.read_csv("orig_data.csv");
df = pd.read_csv("dataset.csv").dropna().sample(n=100000, replace=False,
                                                random_state=seed, axis=0)  # title length scaled
# only drops 4 columns

# these are the initial preprocessed features
X = df.loc[:, df.columns != 'views']
y = np.log10(df['views'])  # this is the output we want to predict
# y = df['views']  # this is the output we want to predict

# splitting into train and test
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed)


print("preparing training:")
prepData = prepareTrainData(x_train)
x_train = prepData[0]

print("preparing test:")
x_test = prepareTestData(x_test, prepData[1])

# doLinearRegression(x_train, y_train, x_test, y_test)
# doSGDRegression(x_train, y_train, x_test, y_test)
# doRidgeRegression(x_train, y_train, x_test, y_test)
# doRidgeCV(x_train, y_train, x_test, y_test)
# doKNNRegression(x_train, y_train, x_test, y_test)
# dKNNGridSearch(x_train, y_train, x_test, y_test)
doRegressionTree(x_train, y_train, x_test, y_test)
# doNeuralNetwork(x_train, y_train, x_test, y_test)

# BEST SO FAR

# RIDGE
# Time to train model: 7 min
# R^2: 0.58
# 1.31

# Time to train model: 13.21 min
# R^2: 0.61
# RMSE: 1.27


# TREE
# Time to train model: 13.21 min
# R^2: 0.7
# RMSE: 1.27
