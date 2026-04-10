# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 13:05:15 2022

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




# Autoregression
def myAutoReg(y,n,c,phi,mu,sigma):
    return c + np.random.normal(mu,sigma) + phi*y

mu = 0
sigma = 2
c = 0
t = np.arange(100)
fig = plt.figure(figsize=(20, 14))

phi=0
y = np.zeros(100)
y[0] = np.random.normal(mu,sigma)
for i in np.arange(99)+1:
    y[i] = myAutoReg(y[i-1],i,c,phi,mu,sigma)
ax1 = plt.subplot(311)
plt.plot(t, y,'k',linewidth=3.5)
plt.xlim(0, 100)
plt.ylim(-10, 10)
plt.ylabel('$y$(t)',fontsize=25)
plt.tick_params('x', labelbottom=False)
plt.yticks(fontsize=12)
ax1.xaxis.grid(linewidth=2)
ax1.yaxis.grid(linewidth=1)
ax1.legend(['$\phi=0$'],fontsize=20)

phi=0.5
y = np.zeros(100)
y[0] = np.random.normal(mu,sigma)
for i in np.arange(99)+1:
    y[i] = myAutoReg(y[i-1],i,c,phi,mu,sigma)
ax2 = plt.subplot(312)
plt.plot(t, y,'k',linewidth=3.5)
plt.xlim(0, 100)
plt.ylim(-10, 10)
plt.ylabel('$y$(t)',fontsize=25)
plt.tick_params('x', labelbottom=False)
plt.yticks(fontsize=12)
ax2.xaxis.grid(linewidth=2)
ax2.yaxis.grid(linewidth=1)
ax2.legend(['$\phi=0.5$'],fontsize=20)

phi=0.9
y = np.zeros(100)
y[0] = np.random.normal(mu,sigma)
for i in np.arange(99)+1:
    y[i] = myAutoReg(y[i-1],i,c,phi,mu,sigma)
ax3 = plt.subplot(313)
plt.plot(t, y,'k',linewidth=3.5)
plt.xlim(0, 100)
plt.ylim(-10, 10)
plt.ylabel('$y$(t)',fontsize=25)
plt.xlabel('t',fontsize=25)
plt.tick_params('x', labelbottom=False)
plt.yticks(fontsize=12)
ax3.xaxis.grid(linewidth=2)
ax3.yaxis.grid(linewidth=1)
ax3.legend(['$\phi=0.9$'],fontsize=20)
plt.grid(True)
# plt.savefig('AR_1_c0.png')
plt.show()
