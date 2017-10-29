#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 13:52:26 2017
RANDOM FOREST REGRESION
@author: devaraju
"""
# Dataset for this model is Position_Salaries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv', sep = ',')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators = 500, random_state = 0)
reg.fit(x, y)

y_pred = reg.predict(6.5)

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, reg.predict(x_grid), color = 'green')
plt.title('Random Forest Regression')
plt.xlabel('position level')
plt.ylabel('salary in dollars')
plt.show()
