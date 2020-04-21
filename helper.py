# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import time
import re

seed = 123
_verbose= False

def tokenizer(text):
    if text:
        result = re.findall('[a-z0-9]{2,}', text.lower())
    else:
        result = []
    return result

def vectorize(df, prop, vectorizer, train=True,prints = False):
    
    if(train):
        start = time.time()
        tfidf = vectorizer.fit_transform(df[prop])
        end = time.time()
        print('Time to train %s vectorizer and transform training text: %0.2fs' % (prop,(end - start)))

        if(prints):
            print('%s\n'%(sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])))
            
    else:
        start = time.time()
        tfidf = vectorizer.transform(df[prop])
        end = time.time()
        print('Time to transform testing %s text: %0.2fs' % (prop,(end - start)))
        
    return (tfidf,vectorizer)

def getFeatures(df, vectorizers, vectorized):
    df = df.iloc[:,4:]
    for vectorizer, vector in zip(vectorizers, vectorized):
        df1 = pd.DataFrame(vector.toarray(), index=df.index, columns=vectorizer.get_feature_names())
        df = pd.concat([df,df1],axis=1)    
        print(df.shape)
        
    return df



# NOTE: Channel title barely affecting model
def prepareTrainData(df):
    
    start = time.time()
    
    # min_df = 0.0007 is the best => the minimum frequency percentage to mark the word/clause as valuable
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
        max_features=20000
        )
    
    channel_vectorizer = TfidfVectorizer(
        tokenizer=tokenizer, 
        token_pattern=r'\w{2,}', 
        # min_df=0.003, 
        # max_df=0.8, 
        smooth_idf=False, 
        sublinear_tf=False, 
        norm=None, 
        analyzer='word',
        # strip_accents='unicode',
        ngram_range=(1, 3),
        max_features=150
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
        max_features=20000
        )
    
    
    vectorizers = [
        title_vectorizer,
        channel_vectorizer, 
        tags_vectorizer
        ]
    
    # vectorize the features that need vectorizing
    print("vectorizing train title")
    title_tfidf = vectorize(df,"title",title_vectorizer,prints=_verbose)[0]
    print("vectorizing train channel_title")
    channel_tfidf = vectorize(df,"channel_title",channel_vectorizer,prints=_verbose)[0]
    print("vectorizing train tags")
    tags_tfidf = vectorize(df,"tags",tags_vectorizer,prints=_verbose)[0]
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
    title_tfidf = vectorize(df,"title",vectorizers[0], train=False)[0]
    print("vectorizing test channel_title")
    channel_tfidf = vectorize(df,"channel_title",vectorizers[1], train=False)[0]
    print("vectorizing test tags")
    tags_tfidf = vectorize(df,"tags",vectorizers[2], train=False)[0]
    vectorized = [
        title_tfidf,
        channel_tfidf,
        tags_tfidf]
    
    df = getFeatures(df, vectorizers, vectorized)
    
    return df
    

def doLinearRegression(x_train, y_train, x_test, y_test):
    print("training model:")
    start = time.time()
    model = LinearRegression(n_jobs=-1).fit(x_train, y_train)
    end = time.time()
    print("R^2: %0.2f" % model.score(x_train,y_train))    
    print('Time to train model: %0.2f min' % ((end - start)/60))
     
    
    print("testing model:")
    y_pred = model.predict(x_test)
    print("RMSE: %0.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
    
    return model


# BEST R^2 = 0.61
def doRidgeRegression(x_train, y_train, x_test, y_test): 
    print("training model:")
    start = time.time()
    model = Ridge(alpha=1, random_state = seed).fit(x_train, y_train)
    end = time.time()
    print('Time to train model: %0.2f min' % ((end - start)/60))
    print("R^2: %0.2f" % model.score(x_train,y_train))      
    
    print("testing model:")
    y_pred = model.predict(x_test)
    print("RMSE: %0.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
    
    return model
    

def doSGDRegression(x_train, y_train, x_test, y_test):     
    
    model = SGDRegressor(loss='squared_loss', penalty='l2', random_state=seed, max_iter=5)
    params = {'penalty':['none','l2','l1'],
              'alpha':[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]}
    gs = GridSearchCV(
        make_pipeline(prepareTrainData(), model),
        # estimator=model,
        param_grid=params,
        scoring='mean_squared_error',
        n_jobs=3,
        cv=5,
        verbose=3
        )
    
    start = time.time()
    gs.fit(x_train, y_train)
    end = time.time()
    print('Time to train model: %0.2fmin' % ((end - start)/60))
    model = gs.best_estimator_
    print(gs.best_params_)
    print(gs.best_score_)

    print("testing model:")
    y_pred = model.predict(x_test)
    print("RMSE: %0.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
