# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn import neighbors
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import re

seed = 123
_verbose = False


def tokenizer(text):
    if text:
        result = re.findall('[a-z0-9]{2,}', text.lower())
    else:
        result = []
    return result


def vectorize(df, prop, vectorizer, train=True, prints=False):

    if(train):
        start = time.time()
        tfidf = vectorizer.fit_transform(df[prop])
        end = time.time()
        print('Time to train %s vectorizer and transform training text: %0.2fs' % (
            prop, (end - start)))

        if(prints):
            print(
                '%s\n' % (sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])))

    else:
        start = time.time()
        tfidf = vectorizer.transform(df[prop])
        end = time.time()
        print('Time to transform testing %s text: %0.2fs' %
              (prop, (end - start)))

    return (tfidf, vectorizer)


def getFeatures(df, vectorizers, vectorized):
    df = df.iloc[:, 3:]
    for vectorizer, vector in zip(vectorizers, vectorized):
        df1 = pd.DataFrame(vector.toarray(), index=df.index,
                           columns=vectorizer.get_feature_names())
        df = pd.concat([df, df1], axis=1)
        print(df.shape)

    return df


# NOTE: Channel title barely affecting model
def prepareTrainData(df):

    start = time.time()

    # min_df = 0.0005 is the best => the minimum frequency percentage to mark the word/clause as valuable
    # max_df => the maximum frequency percentage to mark the word/clause as valuable

    # create vectorizers
    title_vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        token_pattern=r'\w{2,}',
        min_df=0.0005,
        max_df=0.6,
        smooth_idf=False,
        sublinear_tf=False,
        norm=None,
        analyzer='word',
        # strip_accents='unicode',
        ngram_range=(1, 4),
        # max_features=20000
    )

    channel_vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        token_pattern=r'\w{2,}',
        min_df=0.002,
        # max_df=0.8,
        smooth_idf=False,
        sublinear_tf=False,
        norm=None,
        analyzer='word',
        # strip_accents='unicode',
        ngram_range=(1, 3),
        max_features=100
    )

    tags_vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        token_pattern=r'\w{2,}',
        min_df=0.0005,
        max_df=0.6,
        smooth_idf=False,
        sublinear_tf=False,
        norm=None,
        analyzer='word',
        # strip_accents='unicode',
        ngram_range=(1, 3),
        # max_features=20000
    )

    vectorizers = [
        title_vectorizer,
        channel_vectorizer,
        tags_vectorizer
    ]

    # vectorize the features that need vectorizing
    print("vectorizing train title:")
    title_tfidf = vectorize(df, "title", title_vectorizer, prints=_verbose)[0]
    print("vectorizing train channel_title:")
    channel_tfidf = vectorize(
        df, "channel_title", channel_vectorizer, prints=_verbose)[0]
    print("vectorizing train tags:")
    tags_tfidf = vectorize(df, "tags", tags_vectorizer, prints=_verbose)[0]
    vectorized = [
        title_tfidf,
        channel_tfidf,
        tags_tfidf
    ]

    df = getFeatures(df, vectorizers, vectorized)

    end = time.time()

    print('Time to mine text data: %0.2f min' % ((end - start)/60))

    return (df, vectorizers)


def prepareTestData(df, vectorizers):
    # vectorize the features that need vectorizing
    print("vectorizing test title")
    title_tfidf = vectorize(df, "title", vectorizers[0], train=False)[0]
    print("vectorizing test channel_title")
    channel_tfidf = vectorize(
        df, "channel_title", vectorizers[1], train=False)[0]
    print("vectorizing test tags")
    tags_tfidf = vectorize(df, "tags", vectorizers[2], train=False)[0]
    vectorized = [
        title_tfidf,
        channel_tfidf,
        tags_tfidf
    ]

    df = getFeatures(df, vectorizers, vectorized)

    return df


def doLinearRegression(x_train, y_train, x_test, y_test):
    print("training model:")
    start = time.time()
    model = LinearRegression(n_jobs=-1).fit(x_train, y_train)
    end = time.time()
    print("R^2: %0.2f" % model.score(x_train, y_train))
    print('Time to train model: %0.2f min' % ((end - start)/60))

    print("testing model:")
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("MAE: {:.3f}".format(mae))
    print("RMSE: %0.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))

    return model


def doRidgeRegression(x_train, y_train, x_test, y_test):

    model = Ridge(alpha=1, random_state=seed)
    print("training model:")

    start = time.time()
    model.fit(x_train, y_train)
    end = time.time()

    print('Time to train model: %0.2fmin' % ((end - start)/60))
    print("R^2: %0.2f" % model.score(x_train, y_train))

    print("testing model:")
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("MAE: {:.3f}".format(mae))
    print("RMSE: %0.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))

    return model


def doRidgeCV(x_train, y_train, x_test, y_test):
    model = Ridge(alpha=1, random_state=seed)

    params = {
        'alpha': [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]}
    gs = GridSearchCV(estimator=model,
                      param_grid=params,
                      scoring='neg_mean_squared_error',
                      n_jobs=3,
                      cv=5,
                      verbose=3)

    print("training model:")
    start = time.time()
    gs.fit(x_train, y_train)
    end = time.time()
    print('Time to train model: %0.2fmin' % ((end - start)/60))
    model = gs.best_estimator_
    print(gs.best_params_)
    print(gs.best_score_)
    print("R^2: %0.2f" % model.score(x_train, y_train))

    print("testing model:")
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("MAE: {:.3f}".format(mae))
    print("RMSE: %0.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))

    return model


# RESULTS
# Time to train model: 11.42min
# {'alpha': 0.0002, 'penalty': 'l1'}
# -0.40845238240915027
# testing model:
# RMSE: 0.63
def doSGDRegression(x_train, y_train, x_test, y_test, cv=False):

    model = SGDRegressor(loss='squared_loss', penalty='l2',
                         random_state=seed)
    params = {'penalty': ['none', 'l2', 'l1'],
              'alpha': [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]}
    gs = GridSearchCV(estimator=model,
                      param_grid=params,
                      scoring='neg_mean_squared_error',
                      n_jobs=3,
                      cv=5,
                      verbose=3)
    start = time.time()
    gs.fit(x_train, y_train)
    end = time.time()
    print('Time to train model: %0.2fmin' % ((end - start)/60))
    model = gs.best_estimator_
    print(gs.best_params_)
    print(gs.best_score_)
    print("R^2: %0.2f" % model.score(x_train, y_train))

    print("testing model:")
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("MAE: {:.3f}".format(mae))
    print("RMSE: %0.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))

    return model


def doKNNRegression(x_train, y_train, x_test, y_test):
    rmse_val = []  # to store rmse values for different k
    mae_val = []  # to store the mae values for different k
    for K in range(1, 11):
        print("K =", K)

        model = neighbors.KNeighborsRegressor(n_neighbors=K)

        print("fitting the model")
        start = time.time()
        model.fit(x_train, y_train)
        end = time.time()
        print('Time to train model: %0.2f min' % ((end - start)/60))
        print("R^2: %0.2f" % model.score(x_train, y_train))

        print("calculating error: ")
        error = sqrt(mean_squared_error(y_test, y_pred))  # calculate rmse
        rmse_val.append(error)  # store rmse values
        print('RMSE value for k = ', K, 'is: ', error)
        mae = mean_absolute_error(y_test, y_pred)
        mae_val.append(mae)
        print('MAE value for k = ', K, 'is: ', mae)
    curve_rmse = pd.DataFrame(rmse_val)  # elbow curve
    curve_rmse.plot()

    curve_mae = pd.DataFrame(mae_val)
    curve_mae.plot()

    return model


def doKNNGridSearch(x_train, y_train, x_test, y_test):
    params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9],
              'leaf_size': [1, 3, 5],
              'algorithm': ['auto', 'kd_tree', 'ball_tree', 'brute'],
              'metric': ['minkowski', 'euclidean'],
              'n_jobs': [-1]}

    model = neighbors.KNeighborsRegressor()

    gs = GridSearchCV(estimator=model,
                      param_grid=params,
                      scoring='neg_mean_squared_error',
                      n_jobs=3,
                      cv=5,
                      verbose=3)
    start = time.time()
    gs.fit(x_train, y_train)
    end = time.time()
    print('Time to train model: %0.2f min' % ((end - start)/60))
    model = gs.best_estimator_
    print(gs.best_score_)
    print(gs.best_params_)
    print("R^2: %0.2f" % model.score(x_train, y_train))

    # cross validation
    r2_scores = cross_val_score(
        gs.best_estimator_, x_train, y_train, cv=KFold(n_splits=10))
    mse_scores = cross_val_score(gs.best_estimator_, x_train, y_train, cv=KFold(
        n_splits=10), scoring='neg_mean_squared_error')
    print("R^2 for each fold")
    print(r2_scores)
    print("MSE for each fold")
    print(mse_scores)
    print("avg R-squared::{:.3f}".format(np.mean(r2_scores)))
    print("MSE::{:.3f}".format(np.mean(mse_scores)))

    print("testing model:")
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("MAE: {:.3f}".format(mae))
    print("RMSE: %0.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))


def doRegressionTree(x_train, y_train, x_test, y_test):
    print("training model:")
    start = time.time()
    model = DecisionTreeRegressor(min_samples_split=4, random_state=seed).fit(
        x_train, y_train)  # fit the model
    end = time.time()
    print('Time to train model: %0.2f min' % ((end - start)/60))
    print("R^2: %0.2f" % model.score(x_train, y_train))

    print("testing model:")
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("MAE: {:.3f}".format(mae))
    print("RMSE: %0.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))

    print("depth: ", model.get_depth())

    return model


def doRegressionTreeGridSearch(x_train, y_train, x_test, y_test):
    # Hyperparameter tuning using GridSearch
    print("Hyperparameter tuning using GridSearch")
    regressor = DecisionTreeRegressor(random_state=123)
    param_grid = {"criterion": ["mse", "mae"],
                  "min_samples_split": [10, 20, 40],
                  "max_depth": [50, 100, 200],
                  "min_samples_leaf": [20, 40, 100],
                  "max_leaf_nodes": [5, 20, 100],
                  }
    grid_cv_dtm = GridSearchCV(regressor, param_grid, cv=5)
    grid_cv_dtm.fit(x_train, y_train)
    print("R-Squared::{}".format(grid_cv_dtm.best_score_))
    print("Best Hyperparameters::\n{}".format(grid_cv_dtm.best_params_))
    df = pd.DataFrame(data=grid_cv_dtm.cv_results_)
    print(df.head())

    fig, ax = plt.subplots()
    sns.pointplot(data=df[['mean_test_score',
                           'param_max_leaf_nodes',
                           'param_max_depth']],
                  y='mean_test_score', x='param_max_depth',
                  hue='param_max_leaf_nodes', ax=ax)
    ax.set(title="Effect of Depth and Leaf Nodes on Model Performance")
    # Evaluating training model
    print("Evaluating the training model")
    predicted = grid_cv_dtm.best_estimator_.predict(x_train)
    y_train = np.array(y_train)
    residuals = y_train.flatten()-predicted

    fig, ax = plt.subplots()
    ax.scatter(y_train.flatten(), residuals)
    ax.axhline(lw=2, color='black')
    ax.set_xlabel('Observed')
    ax.set_ylabel('Residual')
    plt.show()

    # Checking the training model scores and cross validating (KFold cv)
    print("Checking the training model scores")
    r2_scores = cross_val_score(
        grid_cv_dtm.best_estimator_, x_train, y_train, cv=KFold(n_splits=10))
    mse_scores = cross_val_score(grid_cv_dtm.best_estimator_, x_train, y_train, cv=KFold(
        n_splits=10), scoring='neg_mean_squared_error')
    print("R^2 for each fold")
    print(r2_scores)
    print("MSE for each fold")
    print(mse_scores)
    print("avg R-squared::{:.3f}".format(np.mean(r2_scores)))
    print("MSE::{:.3f}".format(np.mean(mse_scores)))

    # Test dataset evaluation
    print("Test dataset evaluation")
    best_dtm_model = grid_cv_dtm.best_estimator_
    y_pred = best_dtm_model.predict(x_test)
    y_test = np.array(y_test)
    residuals = y_test.flatten() - y_pred
    r2_score = best_dtm_model.score(x_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("R-squared:{:.3f}".format(r2_score))
    print("MAE: {:.3f}".format(mae))
    print("MSE: %.2f" % metrics.mean_squared_error(y_test, y_pred))

    return grid_cv_dtm


def doNeuralNetwork(x_train, y_train, x_test, y_test):

    model = MLPRegressor(
        hidden_layer_sizes=(100,),
        alpha=0.0001,
        max_iter=1000,
        # n_iter_no_change=20,
        solver='adam',  # try lbfgs
        learning_rate='adaptive',  # try adaptive
        learning_rate_init=0.001,
        random_state=seed,
        verbose=True)

    print("training model:")
    start = time.time()
    model.fit(x_train, y_train)
    end = time.time()
    print('Time to train model: %0.2f min' % ((end - start)/60))
    print("R^2: %0.2f" % model.score(x_train, y_train))

    print("testing model:")
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("MAE: {:.3f}".format(mae))
    print("RMSE: %0.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))

    return model


def doNeuralCV(x_train, y_train, x_test, y_test):

    estimator = MLPRegressor(
        hidden_layer_sizes=(100,),
        alpha=0.0001,
        max_iter=1000,
        # n_iter_no_change=20,
        solver='adam',  # try lbfgs
        learning_rate='adaptive',  # try adaptive
        learning_rate_init=0.001,
        random_state=seed,
        verbose=True)

    params = {
        'solver': ['adam', 'lbfgs', 'sgd'],
        'learning_rate': ['constant', 'adaptive'],
        'alpha': [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1],
        'max_iter': [200, 500, 1000]
    }

    model = GridSearchCV(
        estimator=estimator,
        param_grid=params,
        scoring='neg_mean_squared_error',
        n_jobs=3,
        cv=5,
        verbose=True
    )

    print("training model:")
    start = time.time()
    model.fit(x_train, y_train)
    end = time.time()
    print('Time to train model: %0.2f min' % ((end - start)/60))
    print("R^2: %0.2f" % model.score(x_train, y_train))

    print("testing model:")
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("MAE: {:.3f}".format(mae))
    print("RMSE: %0.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))

    return model

# using adam on small
# Iteration 64, loss = 0.06638185
# Iteration 65, loss = 0.06426020
# Iteration 66, loss = 0.06538583
# Training loss did not improve more than tol=0.000100 for 10 consecutive epochs. Stopping.
# Time to train model: 3.23 min
# R^2: 0.84
# testing model:
# RMSE: 1.36

# using lbfgs on small
# Time to train model: 2.37 min
# R^2: 0.95
# testing model:
# RMSE: 1.16
