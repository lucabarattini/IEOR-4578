# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 16:17:23 2022

@author: waseem
"""

import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acovf
from matplotlib.ticker import FormatStrFormatter
import datetime as dt
from scipy import signal
from scipy.fft import fft, ifft
from matplotlib.ticker import AutoMinorLocator
import math
from statsmodels.formula.api import ols
from statsmodels.graphics.api import interaction_plot, abline_plot
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
import scipy
from scipy import linalg


np.random.seed(42)

# Axes limits and knots
x_min, x_knot_1, x_knot_2, x_max = -1, 1.5, 4.5, 7


# Points generated from a cosine function with normal noise
x_true = np.linspace(x_min, x_max,50)
y_true = np.cos(x_true)
y_obs = y_true + np.random.normal(scale=0.5, size=len(x_true))

fig = plt.figure(figsize=(20, 14))
plt.plot(x_true, y_true, linewidth=3, c='gray')
plt.plot(x_true, y_obs, 'o')
# Plot knots
plt.axvline(x=x_knot_1, c='gray', ls='--')
plt.axvline(x=x_knot_2, c='gray', ls='--')
plt.xlabel('x',fontsize=25)
plt.ylabel('y',fontsize=25)
plt.legend(['Original signal','Noisy observations'],fontsize=25)
# plt.savefig('Basis_1.png')
plt.show()


###############################
# Begining of titting piecewise functions to the data
# Get x-y coordinates per region
x_region_1 = x_true[x_true <= x_knot_1]
x_region_2 = x_true[(x_knot_1 < x_true) & (x_true < x_knot_2)]
x_region_3 = x_true[x_true >= x_knot_2]
y_region_1 = y_true[x_true <= x_knot_1]
y_region_2 = y_true[(x_knot_1 < x_true) & (x_true < x_knot_2)]
y_region_3 = y_true[x_true >= x_knot_2]

# Piecewise constant fit ------------------------------------------------
# Plot cosine line and noisy data
fig = plt.figure(figsize=(20, 14))
plt.plot(x_true, y_true, linewidth=3, c='gray')
plt.plot(x_true, y_obs, 'o')
# Plot piecewise constant fits
plt.axhline(y=y_region_1.mean(), c='darkorange', xmin=0, xmax=0.33)
plt.axhline(y=y_region_2.mean(), c='darkorange', xmin=0.33, xmax=0.66)
plt.axhline(y=y_region_3.mean(), c='darkorange', xmin=0.66, xmax=1)
# Plot knots
plt.axvline(x=x_knot_1, c='gray', ls='--')
plt.axvline(x=x_knot_2, c='gray', ls='--')
plt.xlabel('x',fontsize=25)
plt.ylabel('y',fontsize=25)
plt.legend(['Original signal','Noisy observations','Constant mean fit'],fontsize=25)
# plt.savefig('Basis_2.png')
plt.show()


# Piecewise linear fit --------------------------------------------------
# Calculate OLS coefficients from regression anatomy
beta_region_1 = ((y_region_1 - y_region_1.mean()).dot(x_region_1) / 
                (x_region_1**2).sum())
beta_region_2 = ((y_region_2 - y_region_2.mean()).dot(x_region_2) / 
                (x_region_2**2).sum())
beta_region_3 = ((y_region_3 - y_region_3.mean()).dot(x_region_3) / 
                (x_region_3**2).sum())

# Calculate regression fitted values
y_hat_region_1 = beta_region_1 * x_region_1 + y_region_1.mean()
y_hat_region_2 = beta_region_2 * x_region_2 + y_region_2.mean()
y_hat_region_3 = beta_region_3 * x_region_3 + y_region_3.mean()

# Plot cosine line and noisy data
fig = plt.figure(figsize=(20, 14))
plt.plot(x_true, y_true, linewidth=3, c='gray')
plt.plot(x_true, y_obs, 'o')
# Plot piecewise linear fits
plt.plot(x_region_1, y_hat_region_1, c='darkorange')
plt.plot(x_region_2, y_hat_region_2, c='darkorange')
plt.plot(x_region_3, y_hat_region_3, c='darkorange')
# Plot knots
plt.axvline(x=x_knot_1, c='gray', ls='--')
plt.axvline(x=x_knot_2, c='gray', ls='--')
plt.xlabel('x',fontsize=25)
plt.ylabel('y',fontsize=25)
plt.legend(['Original signal','Noisy observations','Linear fit'],fontsize=25)
# plt.savefig('Basis_3.png')
plt.show()


# Continuous Piecewise Linear -------------------------------------------
h1 = np.ones_like(x_true)
h2 = np.copy(x_true)
h3 = np.where(x_true < x_knot_1, 0, x_true - x_knot_1)
h4 = np.where(x_true < x_knot_2, 0, x_true - x_knot_2)
H = np.vstack((h1, h2, h3, h4)).T
# Fit basis expansion via OLS
HH = H.T @ H
beta = scipy.linalg.solve(HH, H.T @ y_obs)
y_hat = H @ beta

# Plot cosine line and noisy data
fig = plt.figure(figsize=(20, 14))
plt.plot(x_true, y_true, linewidth=3,c='gray')
plt.plot(x_true, y_obs, 'o')
plt.plot(x_true, y_hat, c='darkorange')
# Plot knots
plt.axvline(x=x_knot_1, c='gray', ls='--')
plt.axvline(x=x_knot_2, c='gray', ls='--')
plt.xlabel('x',fontsize=25)
plt.ylabel('y',fontsize=25)
plt.legend(['Original signal','Noisy observations','Continuous linear fit'],fontsize=25)
# plt.savefig('Basis_4.png')
plt.show()
# Plot piecewise linear fits
# plt.plot(x_true,h3*beta[2]+beta[0] + beta[1]*x_knot_1, c='red')
# plt.plot(x_true,h3*beta[2]+beta[0] + beta[1]*h2, c='cyan')
# plt.plot(x_true,h3, c='green')



# Construct H
h1 = np.ones_like(x_true)
h2 = np.copy(x_true)
h3 = h2 ** 2
h4 = h2 ** 3
h5 = np.where(x_true < x_knot_1, 0, (x_true - x_knot_1) ** 3)
h6 = np.where(x_true < x_knot_2, 0, (x_true - x_knot_2) ** 3)
H = np.vstack((h1, h2, h3, h4, h5, h6)).T

# Fit basis expansion via OLS
HH = H.T @ H
beta = linalg.solve(HH, H.T @ y_true)
y_hat = H @ beta

# Plot cosine line and noisy data
fig = plt.figure(figsize=(20, 14))
plt.plot(x_true, y_true, linewidth=3,c='gray')
plt.plot(x_true, y_obs, 'o')
plt.plot(x_true, y_hat, c='darkorange')

# Plot knots
plt.axvline(x=x_knot_1, c='gray', ls='--')
plt.axvline(x=x_knot_2, c='gray', ls='--')
plt.xlabel('x',fontsize=25)
plt.ylabel('y',fontsize=25)
plt.legend(['Original signal','Noisy observations','Continuous cubic spline fit'],fontsize=25)
# plt.savefig('Basis_5.png')
plt.show()



###############################
###############################
# Load the Whole Food data
dw = pd.read_excel('WholeFood.xlsx')
y_true = dw['Weekly Sales'].values
y_Obs = dw['Weekly Sales'].values
y_obs = dw['Weekly Sales'].values
x_true = np.arange(len(dw))

dw = pd.read_csv('IPN31152N_Fred_Industrial Production Manufacturing Non-Durable Goods Ice Cream and Frozen Dessert.csv')
y_true = dw['IPN31152N'].values
y_Obs = dw['IPN31152N'].values
y_obs = dw['IPN31152N'].values
x_true = np.arange(len(dw))

