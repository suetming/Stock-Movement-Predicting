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
import datetime

# load all stocks
data_hs_300 = pd.read_csv("./300.csv", dtype=object)
data_stocks = pd.read_csv("./stocks.csv", dtype=object)
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

def save_all_in_one(startDate, endDate, days=10):
    # get all Shanghai and Shenzhen 300 index Stock to csv
    hs300 = np.array(data_hs_300['code'])
    data = fetch_stock_data(hs300, startDate, endDate, days)
    np.savetxt("test.csv", data, delimiter=",", header=','.join([str(i) for i in range(1,5 * days + 1)]))

def save_all_in_industry(name, codes, startDate, endDate, days=10):
    data = fetch_stock_data(codes, startDate, endDate, days)
    np.savetxt('test.' + name + '.' + str(days) + '.csv', data, delimiter=",", header=','.join([str(i) for i in range(1, 5 * days + 1)]))

def save_specific_stock(ticker, startDate, endDate, days=10):
    # save specific stock
    hs300 = np.array(data_hs_300['code'])
    dates = []
    startDate = '2016-12-06'
    dates.append(datetime.date.today())
    endDate = str(dates[0])
    data = fetch_stock_data([ticker], startDate, endDate, days)
    np.savetxt("test." + ticker + ".csv", data, delimiter=",", header=','.join([str(i) for i in range(1,5 * days + 1)]))

if __name__ == '__main__':
    dates = []
    startDate = '2016-12-06'
    dates.append(datetime.date.today())
    endDate = str(dates[0])

    save_all_in_one(startDate, endDate)

    print "prepare fetch industry"

    for c_name in c_names:
        codes = data_stocks[data_stocks['c_name'] == c_name]['code']
        codes_300 = []
        for code in codes:
            s = data_hs_300[data_hs_300['code'] == code]
            if not s.empty:
                codes_300.append(s.values[0][0])
        if len(codes_300) > 10: # we only find the industry more than 10
            print c_name, len(codes_300)
            save_all_in_industry(c_name, codes_300, startDate, endDate, 10)

    save_specific_stock("000001", startDate, endDate)
    save_specific_stock("000002", startDate, endDate)
    save_specific_stock("000568", startDate, endDate)
    save_specific_stock("000625", startDate, endDate)
    save_specific_stock("000651", startDate, endDate)
    save_specific_stock("000876", startDate, endDate)
