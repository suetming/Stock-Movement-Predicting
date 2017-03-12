#!/usr/bin/python
#-*-coding:utf-8 -*-

# Author: suetming <309201678@qq.com>
# store the all stock to csv (open/close/high/low/volume)

import plotly
import numpy as np
import plotly.graph_objs as go

plotly.tools.set_credentials_file(username='suetming', api_key='OpF8S21eU1j4hGmnzY1c')


# create the x and y axixes for our plot
x = ["Ridge",
     "LASSO",
     "Random Forest",
     "GBDT",
     "Ridge-RF",
     "Ridge-RF-LASSO-GBDT"]
y = ["Media",
     "Power",
     "IT",
     "Estate",
     "Materials",
     "Transportation",
     "Financials",
     "Automobile",
     "Biopharmaceutical"]
y.reverse()

# the data from our runs of the 6 models on the 9 datasets
z = [[0.941601,     0.948539,       0.947358,       0.932497,   0.950367,       0.937084,   0.936564,       0.945170,   0.951262],\
     [0.941964,     0.948554,       0.947487,       0.932600,   0.950509,       0.937157,   0.936235,       0.945150,   0.951247],\
     [0.906478,     0.917134,       0.922514,       0.887982,   0.896142,       0.898507,   0.930417,       0.892292,   0.917381], \
     [0.939190,     0.943718,       0.945342,       0.921626,   0.934660,       0.933729,   0.934772,       0.935471,   0.944203], \
     [0.9526214660, 0.955268258,    0.954110676,    0.93632073, 0.9522594940,   0.94395894, 0.948832834,    0.94702756, 0.954939734], \
     [0.951802132,  0.9549889,      0.953371776,    0.936847972, 0.95273992,    0.94347202, 0.948426308,    0.94645516, 0.955034464]]

# plotly url
print str(plotly.plotly.plot([{'x': x,'y': y,'z': np.array(z).T.tolist(),  'type': 'heatmap', 'colorscale': 'Viridis'}])['url'])
