# -*- coding: utf-8 -*-
from helper import *    

# df = pd.read_csv("orig_data.csv");
df = pd.read_csv("dataset.csv"); # title length scaled
df = df.dropna() # only drops 4 columns

X = df.loc[:, df.columns != 'views'] # these are the initial preprocessed features
y = np.log(df['views']) # this is the output we want to predict

# splitting into train and test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)





print("preparing training:")
prepData = prepareTrainData(x_train)
x_train = prepData[0]

print("preparing test:")
x_test = prepareTestData(x_test, prepData[1])

# doLinearRegression(x_train, y_train, x_test, y_test)
# doRidgeRegression(x_train, y_train, x_test, y_test)


# note this is using cross validation so the preparation should be done inside the function
# i'm not sure how to do that so for now it's not like that
# doSGDRegression(x_train, y_train, x_test, y_test)



# BEST SO FAR 

# Time to train model: 7 min
# R^2: 0.58
# 1.31

# Time to train model: 13.21 min
# R^2: 0.61
# RMSE: 1.27

























