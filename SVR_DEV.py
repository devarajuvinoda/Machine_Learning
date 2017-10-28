#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 02:20:21 2017
SUPPORT VECTOR REGRESSION(SVR)
@author: devaraju
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv', sep = ',')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x) 
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
reg = SVR(kernel = 'rbf')
reg.fit(x, y)

y_pred = sc_y.inverse_transform(reg.predict(sc_x.transform(np.array([[6.5]])))) 

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid),1)) 
plt.scatter(x, y,color= 'red')
plt.plot(x, reg.predict(x), color = 'green')
plt.show()