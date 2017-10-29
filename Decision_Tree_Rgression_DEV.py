#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 12:27:23 2017
DECISION TREE REGRESSION
@author: devaraju
"""
# Dataset for this model is Pisition_Salaries.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv', sep = ',')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values

from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(random_state = 0)
reg.fit(x, y)

y_pred = reg.predict(6.5)

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, reg.predict(x_grid), color = 'green')
plt.title('Decision Tree Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()