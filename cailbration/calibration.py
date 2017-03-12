#!/usr/bin/python
#-*-coding:utf-8 -*-

# set some nicer defaults for matplotlib
from matplotlib import rcParams
import sklearn.linear_model as lm
import sys
import numpy as np
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from os.path import basename

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
    sys.stdout.write('.')
    predictions = [[] for model in models]
    newY = []

    for i in range(folds):
        idx = np.arange(i, X.shape[0], folds)
        idx_fit = list(set(range(X.shape[0])) - set(np.arange(i, X.shape[0], folds)))

        for i, model in enumerate(models):
            instance = model.fit(X[idx_fit,:], y[idx_fit])
            predictions[i].extend(list(instance.predict_proba(X[idx,:])[:,1]))

        # put together the reordered newY
        newY = newY + list(y[idx])

    # format everything for return
    newX = np.hstack([np.array(prediction).reshape(len(prediction), 1) for prediction in predictions])
    newY = np.array(newY).reshape(len(newY), 1)
    return newX, newY

def remove_all(axes=None, top=False, right=False, left=True, bottom=True):
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)

    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()

def cplot(clf, xtest, ytest, title = ""):
    """
    taken from https://www.coursehero.com/file/p6vdpkr/Take-a-collection-of-examples-and-compute-the-freshness-probability-for-each/
    """
    prob = clf.predict_proba(xtest)[:, 1]
    outcome = ytest
    data = pd.DataFrame(dict(prob=prob, outcome=outcome))

    bins = np.linspace(0, 1, 20)
    cuts = pd.cut(prob, bins)
    binwidth = bins[1] - bins[0]

    #freshness ratio and number of examples in each bin
    cal = data.groupby(cuts).outcome.agg(['mean', 'count'])
    cal['pmid'] = (bins[:-1] + bins[1:]) / 2
    cal['sig'] = np.sqrt(cal.pmid * (1 - cal.pmid) / cal['count'])

    #the calibration plot
    ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    p = plt.errorbar(cal.pmid, cal['mean'], cal['sig'])
    plt.plot(cal.pmid, cal.pmid, linestyle='--', lw=1, color='k')
    plt.ylabel("Empirical P(Stock increase)")
    plt.ylim((0, 1))
    remove_all(ax)
    plt.title(title)


    ax = plt.subplot2grid((3, 1), (2, 0), sharex=ax)

    plt.bar(left=cal.pmid - binwidth / 2, height=cal['count'],
            width=.95 * (bins[1] - bins[0]),
            fc=p[0].get_color())

    plt.xlabel("Predicted P(Stock increase)")
    remove_all()
    plt.ylabel("Number")

if __name__ == '__main__':
    rcParams['figure.figsize'] = (10, 6)
    rcParams['figure.dpi'] = 150
    rcParams['axes.color_cycle'] = [(0.843, 0.188, 0.122)]
    rcParams['lines.linewidth'] = 2
    rcParams['axes.grid'] = False
    rcParams['axes.facecolor'] = 'white'
    rcParams['font.size'] = 10
    rcParams['patch.edgecolor'] = 'none'

    train = np.array(pd.read_table('hs300.csv', sep = ","))

    columns_we_want = list(chain.from_iterable([[5 * x, 5 * x + 1] for x in range(10)]))[:-1]

    # got only open and close prices
    X = np.array([l/l[0] for l in train[:, columns_we_want]])

     # the stock up or down.
    y = (train[:, 46] > train[:, 45]) + 0

    print X, y
    print len(X)
    # train_size = 103136
    # # ridge
    # ridge_model = lm.LogisticRegression(penalty = "l2", C = 4500)
    # ridge_model.fit(X[:train_size,:], y[:train_size]) # train model
    # cplot(ridge_model, X[train_size:,:], y[train_size:], "Logistic Ridge") # make calibration plot
    # print "ridge.png"
    # plt.savefig('ridge.png')
    #
    # # lasso
    # lasso_model = lm.LogisticRegression(penalty = "l1", C = 1000)
    # lasso_model.fit(X[:train_size,:], y[:train_size]) # train model
    # cplot(lasso_model, X[train_size:,:], y[train_size:], "Logistic LASSO")
    # print "lasso.png"
    # plt.savefig('lasso.png')
    #
    # # random forest
    # rf_model = RandomForestClassifier(n_estimators = 200)
    # rf_model.fit(X[:train_size,:], y[:train_size]) # train model
    # cplot(rf_model, X[train_size:,:], y[train_size:], "Random Forest")
    # print "rf.png"
    # plt.savefig('rf.png')
    #
    # # gbdt
    # gbt_model = GradientBoostingClassifier(n_estimators = 200)
    # gbt_model.fit(X[:train_size,:], y[:train_size]) # train model
    # cplot(gbt_model, X[train_size:,:], y[train_size:], "Gradient Boosted Decision Trees")
    #
    # print "gbdt.png"
    # plt.savefig('gbdt.png')

    # models = [lm.LogisticRegression(penalty='l2', C = 4500),
    #           lm.LogisticRegression(penalty='l1', C = 2500),
    #           RandomForestClassifier(n_estimators = 200),
    #           GradientBoostingClassifier(n_estimators = 200),
    #           ]
    #
    # newX, newY = get_newX_y(models, X, y)
    #
    # print newX, newY

    # all_model_list = lm.LogisticRegression()
    # all_model_list.fit(newX[:train_size,:], newY.reshape(newY.shape[0])[:train_size]) # train model
    #
    # cplot(all_model_list, newX[train_size:,:], newY.reshape(newY.shape[0])[train_size:], "Ridge-lasso-rf-gbdt Blended model")
    # print "ridge-lasso-rf-gbdt.png"
    # plt.savefig('ridge-lasso-rf-gbdt.png')
    #
    # models = [lm.LogisticRegression(penalty='l2', C = 4500),
    #           RandomForestClassifier(n_estimators = 200)]
    #
    # newX, newY = get_newX_y(models, X, y)
    #
    # all_model_list = lm.LogisticRegression()
    # all_model_list.fit(newX[:train_size,:], newY.reshape(newY.shape[0])[:train_size]) # train model
    #
    # cplot(all_model_list, newX[train_size:,:], newY.reshape(newY.shape[0])[train_size:], "Ridge-rf Blended model")
    # print "ridge-rf.png"
    # plt.savefig('ridge-rf.png')
