#!/usr/bin/python
#-*-coding:utf-8 -*-

# Author: suetming <309201678@qq.com>

import sys
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import metrics,preprocessing
from sklearn import cross_validation
# from sklearn.model_selection import cross_val_score
from itertools import chain

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

def model_score(X, y, model = "ridge", dataset = ""):
    # C here is the list of tuning parameters that we want to test
    if model == "ridge":
        C = np.linspace(500, 5000, num = 10)
        models = [lm.LogisticRegression(penalty = "l2", C = c) for c in C]
    elif model == "lasso":
        C = np.linspace(500, 5000, num = 10)
        models = [lm.LogisticRegression(penalty = "l1", C = c) for c in C]
    elif model == "randomforest":
        C = np.linspace(10, 200, num = 20)
        models = [RandomForestClassifier(n_estimators = int(c)) for c in C]
    elif model == "gbt":
        C = np.linspace(10, 200, num = 20)
        models = [GradientBoostingClassifier(n_estimators = int(c)) for c in C]

    # print "calculating cv scores"
    cv_scores = [0] * len(models)
    for i, model_ in enumerate(models):
        cv_scores[i] = np.mean(cross_validation.cross_val_score(model_, X, y, cv=5, scoring = scorer_auc))
        print "%s;%s;(%d/%d);%f;%f" % (dataset, model, i + 1, len(C), C[i], cv_scores[i])

    best = cv_scores.index(max(cv_scores))
    best_model = models[best]
    best_cv = cv_scores[best]
    best_C = C[best]
    print "%s;%s;BEST;%f: %f" % (dataset, model, best_C, best_cv)

def model_score(X, y, model = "ridge", debug=True):
    if model == "ridge":
        C = np.logspace(1, 4, num = 10)
        # C = np.array([1, 5, 10, 50, 100, 500, 1000, 5000])
        models = [lm.LogisticRegression(penalty = "l2", C = c) for c in C]

    if model == "lasso":
        C = np.logspace(1, 4, num = 10)
        # C = np.array([1, 5, 10, 50, 100, 500, 1000, 5000])
        models = [lm.LogisticRegression(penalty = "l1", C = c) for c in C]

    if model == "randomforest":
        C = np.linspace(10, 200, num = 20)
        models = [RandomForestClassifier(n_estimators = int(c)) for c in C]

    if model == "gbt":
        C = np.linspace(10, 200, num = 20)
        models = [GradientBoostingClassifier(n_estimators = int(c)) for c in C]

    cv_scores = [0] * len(models)
    for i, model_ in enumerate(models):
        cv_scores[i] = np.mean(cross_validation.cross_val_score(model_, X, y, cv=5, scoring = scorer_auc))
        if debug:
            print "%s;(%d/%d);%f;%f" % (model, i + 1, len(C), C[i], cv_scores[i])

    best = cv_scores.index(max(cv_scores))
    best_model = models[best]
    best_cv = cv_scores[best]
    best_C = C[best]
    print "%s;BEST;%f;%f" % (model, best_C, best_cv)

if __name__ == '__main__':
    # 8, 9, 11,
    csvs = ['hs300_5.csv', 'hs300_6.csv','hs300_7.csv', 'hs300_8.csv', 'hs300_9.csv', 'hs300_10.csv', 'hs300_11.csv', 'hs300_12.csv', 'hs300_13.csv', 'hs300_14.csv', 'hs300_15.csv', 'hs300_16.csv', 'hs300_17.csv', 'hs300_18.csv', 'hs300_19.csv']
    idx = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    for idx in idx:
        name = 'hs300_' + str(idx) +'.csv'
        print name
        train = np.array(pd.read_table(name, sep = ","))

        columns_we_want = list(chain.from_iterable([[5 * x, 5 * x + 1] for x in range(idx)]))[:-1]

        # got open and close prices and normalize
        X = np.array([l/l[0] for l in train[:, columns_we_want]])

        # the stock up or down.
        y = (train[:, 5 * idx - 4] > train[:, 5 * idx - 5]) + 0

        print "calculating cv scores"
        print "model;step;C;CV"
        # models = ['ridge', 'lasso', 'randomforest', 'gbt']
        models = ['ridge', 'lasso']
        for model in models:
            model_score(X, y, model, debug=True)
