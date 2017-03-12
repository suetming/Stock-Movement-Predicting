#!/usr/bin/python
#-*-coding:utf-8 -*-

# Author: suetming <309201678@qq.com>
from __future__ import unicode_literals

import re
import os
import csv
import pandas as pd
import tushare as ts
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt


if __name__ == '__main__':
    name = {
        u"交通运输" :  u"Transportation",
        u"传媒娱乐":  u"Media entertainment",
        u"建筑材料" : u"Architect trades",
        u"电力行业" : u"Power industry",
        u"电子信息" : u"Electronic and information technology",
        u"房地产" : u"Real estate industry",
        u"金融行业" : u"Financials",
        u"汽车制造" : u"Automobile manufacturing",
        u"生物制药" : u"Biopharmaceutical"
    }
    # all HUSHEN 300 stocks prices figure
    train = np.array(pd.read_table('hs300.csv', sep = ","))
    #
    plt.scatter(train[:, 46] / train[:, 45], train[:, 45] / train[:, 41])
    plt.xlim((.8,1.2)); plt.ylim((.8,1.2))
    plt.xlabel("opening[n]/closing[n]"); plt.ylabel("opening[n]/closing[n - 1]")
    plt.show()

    # plt.scatter(train[:, 46] / train[:, 45], train[:, 49] / train[:, 44]) # volume[n] / volume[n-1] - ClOSING[n] /  OPENING[n]
    # plt.xlim((.8,1.2)); plt.ylim((.8,1.2))
    # plt.xlabel("opening[n]/closing[n]"); plt.ylabel("volume[n]/volume[n-1]")
    # plt.show()

    # plt.scatter(train[:, 46] / train[:, 45], train[:, 47] / train[:, 42]) # volume[n] / volume[n-1] - ClOSING[n] /  OPENING[n]
    # plt.xlim((.8,1.2)); plt.ylim((.8,1.2))
    # plt.xlabel("opening[n]/closing[n]"); plt.ylabel("high[n]/high[n-1]")
    # plt.show()

    # different industry figure in HUSHEN 300 stocks
    i = 0
    list = os.listdir("data/300/industry")
    count = len(list)

    fig = plt.figure(figsize=(2, count /  2 + 1))

    c = (count /  2)
    for file in list:
        if file.endswith(".csv"):
            train = np.array(pd.read_table("data/300/industry/" + file, sep = ","))
            ax = fig.add_subplot(5, 2, i + 1)
            ax.scatter(train[:, 46] / train[:, 45], train[:, 45] / train[:, 41])
            ax.set_xlim((.8,1.2)); ax.set_ylim((.8,1.2))

            # ax.set_ylabel('CLOSING10/OPENING10', fontsize=9)
            # ax.set_xlabel('OPENING10/CLOSING9', fontsize=9, labelpad=0)
            ax.set_title(name.get(os.path.splitext(file)[0]), fontsize= 9)
            i = i + 1
    plt.show()
