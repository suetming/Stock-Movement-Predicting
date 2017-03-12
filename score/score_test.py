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
from sklearn.metrics import roc_curve, auc, roc_auc_score
from itertools import chain

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

    cv_scores = [0] * len(models)
    for i, model_ in enumerate(models):
        cv_scores[i] = np.mean(cross_validation.cross_val_score(model_, X, y, cv=5, scoring = scorer_auc))
        print "%s;%s;(%d/%d);%f;%f" % (dataset, model, i + 1, len(C), C[i], cv_scores[i])

    best = cv_scores.index(max(cv_scores))
    best_model = models[best]
    best_cv = cv_scores[best]
    best_C = C[best]
    print "%s;%s;BEST;%f: %f" % (dataset, model, best_C, best_cv)
    return (dataset, model, best_C, best_cv)

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
    name = '../market/train.csv'
    train = np.array(pd.read_table(name, sep = ","))
    columns_we_want = list(chain.from_iterable([[5 * x, 5 * x + 1] for x in range(idx)]))[:-1]

    # got open and close prices and normalize
    X = np.array([l/l[0] for l in train[:, columns_we_want]])

    # the stock up or down.
    y = (train[:, 5 * idx - 4] > train[:, 5 * idx - 5]) + 0

    print "calculating cv scores"
    print "model;step;C;CV"
    models = ['ridge', 'lasso', 'randomforest', 'gbt']
    for model in models:
        model_score(X, y, model, debug=True)
