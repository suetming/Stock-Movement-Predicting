#!/usr/bin/python
#-*-coding:utf-8 -*-

# Author: suetming <309201678@qq.com>
# store the all stock to csv (open/close/high/low/volume)

import re
import os
import csv
import pandas as pd
import tushare as ts
import numpy as np
import matplotlib.pyplot as plt

# load all stocks
data_hs_300 = pd.read_csv("./data/300.csv", dtype=object)
data_stocks = pd.read_csv("./data/stocks.csv", dtype=object)
c_names = data_stocks['c_name'].unique()

# data includes open, close, high, low, volume and code.
def fetch_stock_data(tickers, start, end, days=10):
    results = []
    for ticker in tickers:
        print ticker
        try:
            data = np.array(ts.get_k_data(ticker, start, end).drop(['code', 'date'], axis = 1))[::-1]
            result = []
            for i in range(days):
                result.append(data[i:(i+data.shape[0]- days + 1),:])
        except:
            pass
        results.append(np.hstack(result))
    return np.vstack(results)

# all stock data to csv
def save_all_stock():
    for c_name in c_names:
        codes = data_stocks[data_stocks['c_name'] == c_name]['code']
        for code in codes:
            ts.get_k_data(code).to_csv('data/' + c_name + '/' + code + '.csv', encoding='utf-8')
            print code

def save_all_in_one(days=10):
    # get all Shanghai and Shenzhen 300 index Stock to csv
    hs300 = np.array(data_hs_300['code'])

    # for stock in hs300:
    #     ts.get_k_data(stock).to_csv('data/300/' + stock + '.csv', encoding='utf-8')
    #     print stock
    data = fetch_stock_data(hs300, '2015-01-01', '2016-12-05', days)
    np.savetxt("hs300" + '_' + str(days) + ".csv", data, delimiter=",", header=','.join([str(i) for i in range(1,5 * days + 1)]))

def save_all_in_industry(name, codes, days=10):
    data = fetch_stock_data(codes, '2015-01-01', '2016-12-05', days)
    np.savetxt('data/300/industry/' + name + '_' + str(days) + '.csv', data, delimiter=",", header=','.join([str(i) for i in range(1, 5 * days + 1)]))

if __name__ == '__main__':
    for i in range(5, 20):
        save_all_in_one(i)
        for c_name in c_names:
            codes = data_stocks[data_stocks['c_name'] == c_name]['code']
            codes_300 = []
            for code in codes:
                s = data_hs_300[data_hs_300['code'] == code]
                if not s.empty:
                    codes_300.append(s.values[0][0])
            if len(codes_300) > 10: # we only find the industry more than 10
                print c_name, len(codes_300)
                save_all_in_industry(c_name, codes_300, i)
