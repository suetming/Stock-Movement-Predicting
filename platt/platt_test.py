#!/usr/bin/python
#-*-coding:utf-8 -*-

# Author: suetming <309201678@qq.com>
# store the all stock to csv (open/close/high/low/volume)

import re
import os
import csv
import sys
import pandas as pd
import tushare as ts
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import metrics,preprocessing,cross_validation

def tied_rank(x):
    """
    This function is by Ben Hamner and taken from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py
    """
    sorted_x = sorted(zip(x,range(len(x))))
    r = [0 for k in x]
    cur_val = sorted_x[0][0]
    last_rank = 0
    for i in range(len(sorted_x)):
        if cur_val != sorted_x[i][0]:
            cur_val = sorted_x[i][0]
            for j in range(last_rank, i):
                r[sorted_x[j][1]] = float(last_rank+1+i)/2.0
            last_rank = i
        if i==len(sorted_x)-1:
            for j in range(last_rank, i+1):
                r[sorted_x[j][1]] = float(last_rank+i+2)/2.0
    return r

def auc(actual, posterior):
    """
    This function is by Ben Hamner and taken from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py
    """
    r = tied_rank(posterior)
    num_positive = len([0 for x in actual if x==1])
    num_negative = len(actual)-num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if actual[i]==1])
    auc = ((sum_positive - num_positive*(num_positive+1)/2.0) /
           (num_negative*num_positive))
    return auc

def scorer_auc(estimator, X, y):
    predicted = estimator.predict_proba(X)[:,1]
    return auc(y, predicted)

def get_newX_y(models, X, y, folds = 10):
    # this is simply so we know how far the model has progressed
    sys.stdout.write('.')
    predictions = [[] for model in models]
    newY = []

    # for every fold of the data...
    for i in range(folds):
        # find the indices that we want to train and predict
        idx = np.arange(i, X.shape[0], folds)
        idx_fit = list(set(range(X.shape[0])) - set(np.arange(i, X.shape[0], folds)))

        # put together the predictions for each model
        for i, model in enumerate(models):
            instance = model.fit(X[idx_fit,:], y[idx_fit])
            predictions[i].extend(list(instance.predict_proba(X[idx,:])[:,1]))

        # put together the reordered newY
        newY = newY + list(y[idx])

    # format everything for return
    newX = np.hstack([np.array(prediction).reshape(len(prediction), 1) for prediction in predictions])
    newY = np.array(newY).reshape(len(newY), 1)
    return newX, newY

def get_test_data(name):
    test = np.array(pd.read_table('../market/test' + name + ".csv", sep = ","))

    columns_test = list(chain.from_iterable([[5 * x, 5 * x + 1] for x in range(10)]))[:-1]

    # got only open and close prices
    X_test = np.array([l/l[0] for l in test[:, columns_test]])

    # the stock up or down.
    y_test = np.array((test[:, 46] > test[:, 45]) + 0)

    return X_test, y_test

if __name__ == '__main__':
    names = ['ridge', 'lasso', 'randomforest', 'gbdt']
    models = [lm.LogisticRegression(penalty='l2', C = 4500),
              lm.LogisticRegression(penalty='l1', C = 2500),
              RandomForestClassifier(n_estimators = 200),
              GradientBoostingClassifier(n_estimators = 200)]

    train = np.array(pd.read_table('../market/train.csv', sep = ","))

    columns_train = list(chain.from_iterable([[5 * x, 5 * x + 1] for x in range(10)]))[:-1]

    # got only open and close prices
    X_train = np.array([l/l[0] for l in train[:, columns_train]])

    # the stock up or down.
    y_train = np.array((train[:, 46] > train[:, 45]) + 0)

    X_test, y_test = get_test_data("")

    for name, model in zip(names, models):
        model.fit(X_train, y_train)
        # print "all", name, scorer_auc(model, X_test, y_test), 1

    newX_train, newY_train = get_newX_y(models, X_train, y_train)
    # newX_test, newY_test = get_newX_y(models, X_test, y_test)

    # combine model
    all_model_list = lm.LogisticRegression()

    # new model
    all_model_list.fit(newX_train, newY_train.reshape(newY_train.shape[0]))

    # we see what weights the blended model assigns to the probability predictions of each model.
    # print "all", "ridge-lasso-rf-gbdt", scorer_auc(all_model_list, newX_test, newY_test), all_model_list.coef_

    models_rf_ridge = [lm.LogisticRegression(penalty='l2', C = 4500),RandomForestClassifier(n_estimators = 200)]
    newX_train, newY_train = get_newX_y(models_rf_ridge, X_train, y_train)
    # newX_test, newY_test = get_newX_y(models_rf_ridge, X_test, y_test)

    # combine model
    ridge_rf_model_list = lm.LogisticRegression()

    newX_train_rf_ridge, newY_train_rf_ridge = get_newX_y(models_rf_ridge, X_train, y_train)
    # newX_test_rf_ridge, newY_test_rf_ridge = get_newX_y(models_rf_ridge, X_test, y_test)

    # new model
    ridge_rf_model_list.fit(newX_train_rf_ridge, newY_train_rf_ridge.reshape(newY_train_rf_ridge.shape[0]))
    # we see what weights the blended model assigns to the probability predictions of each model.
    # print "all", "ridge-rf", scorer_auc(ridge_rf_model_list, newX_test_rf_ridge, newY_test_rf_ridge), ridge_rf_model_list.coef_

    # stock test
    # industries = ["传媒娱乐", "电力行业", "房地产", "建筑建材", "交通运输", "金融行业", "汽车制造", "生物制药"]
    industries = ["电子信息"]

    for industry in industries:
        X, y = get_test_data("." + industry + ".10")
        for name, model in zip(names, models):
            print industry, name, scorer_auc(model, X, y), 1

        newX, newY = get_newX_y(models, X, y)
        print industry, "ridge-lasso-rf-gbdt", scorer_auc(all_model_list, newX, newY), all_model_list.coef_

        newX_rd, newY_rd = get_newX_y(models_rf_ridge, X, y)
        print industry, "ridge-rf", scorer_auc(ridge_rf_model_list, newX_rd, newY_rd), ridge_rf_model_list.coef_
