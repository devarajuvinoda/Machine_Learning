#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 01:31:12 2017
POLYNOMIAL REGRESSION
@author: devaraju
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv', sep = ',')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2:].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

plt.scatter(x, y, color= 'red')
plt.plot(x, lin_reg.predict(x), color = 'green')
plt.title('LinearRegression')
plt.xlabel('position level')
plt.ylabel('salary in dollars')
plt.show()


x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)))
plt.title('PolynomialRegression')
plt.xlabel('position level')
plt.ylabel('salary in dollars')
plt.show()