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
    This function is taken from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py
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
    This function is taken from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py
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
    predictions = [[] for model in models]
    newY = []

    for i in range(folds):
        idx = np.arange(i, X.shape[0], folds)
        idx_fit = list(set(range(X.shape[0])) - set(np.arange(i, X.shape[0], folds)))

        for i, model in enumerate(models):
            instance = model.fit(X[idx_fit,:], y[idx_fit])
            predictions[i].extend(list(instance.predict_proba(X[idx,:])[:,1]))

        newY = newY + list(y[idx])

    newX = np.hstack([np.array(prediction).reshape(len(prediction), 1) for prediction in predictions])
    newY = np.array(newY).reshape(len(newY), 1)
    return newX, newY

if __name__ == '__main__':
    datasets = os.listdir("data/300/industry")#列出目录下的所有文件和目录
    for dataset in datasets:
        if dataset.endswith(".csv"):
            train = np.array(pd.read_table("data/300/industry/" + dataset, sep = ","))

            columns_we_want = list(chain.from_iterable([[5 * x, 5 * x + 1] for x in range(10)]))[:-1]

            # got only open and close prices
            X = np.array([l/l[0] for l in train[:, columns_we_want]])

             # the stock up or down.
            y = (train[:, 46] > train[:, 45]) + 0

            print dataset

            print 'ridge-lasso-rf-gbdt'

            models = [lm.LogisticRegression(penalty='l2', C = 4500),
                      lm.LogisticRegression(penalty='l1', C = 2500),
                      RandomForestClassifier(n_estimators = 200),
                      GradientBoostingClassifier(n_estimators = 200),
                      ]

            #
            newX, newY = get_newX_y(models, X, y)

            # combine model
            all_model_list = lm.LogisticRegression()


            print cross_validation.cross_val_score(all_model_list, newX, newY.reshape(newY.shape[0]), cv=5, scoring = scorer_auc)

            # new model
            all_model_list.fit(newX, newY.reshape(newY.shape[0]))

            # we see what weights the blended model assigns to the probability predictions of each model.
            print all_model_list.coef_

            print 'ridge-rf'
            models = [lm.LogisticRegression(penalty='l2', C = 5000),
                      RandomForestClassifier(n_estimators = 180),
                      ]
            #
            newX, newY = get_newX_y(models, X, y)

            # combine model
            all_model_list = lm.LogisticRegression()


            print cross_validation.cross_val_score(all_model_list, newX, newY.reshape(newY.shape[0]), cv=5, scoring = scorer_auc)

            # new model
            all_model_list.fit(newX, newY.reshape(newY.shape[0]))

            # we see what weights the blended model assigns to the probability predictions of each model.
            print all_model_list.coef_
