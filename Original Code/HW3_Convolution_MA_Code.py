# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 12:33:37 2022

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


dv = pd.read_excel('datasets/Viscosity.xlsx')
dw = pd.read_excel('datasets/WholeFood.xlsx')
dp = pd.read_excel('datasets/Pharmaceutical.xlsx')


### Convolution and MA
t = np.arange(0, dv.shape[0])
s1 = dv.Viscosity.values
# Convolution
filtered = signal.convolve(s1, [1/5,1/5,1/5,1/5,1/5], mode='same')
# MA
x = pd.DataFrame(s1)
x = np.concatenate( x.rolling(5).mean().values.tolist(), axis=0 )
fig = plt.figure(figsize=(20, 14))

ax1 = plt.subplot(211)
plt.plot(t, s1,'k')
plt.plot(t, filtered,'--k',linewidth=3.5)
plt.xlim(0, 100)
plt.ylim(0, 10)
plt.ylabel('$X$(t)',fontsize=25)
plt.tick_params('x', labelbottom=False)
plt.yticks(fontsize=12)
ax1.xaxis.grid(linewidth=2)
ax1.yaxis.grid(linewidth=1)
ax1.legend(['data','Convolution'],fontsize=20)

ax2 = plt.subplot(212)
plt.plot(t, s1,'k')
plt.plot(t, x,'--k',linewidth=3.5)
plt.xlim(0, 100)
plt.ylim(0, 10)
plt.ylabel('$X$(t)',fontsize=25)
plt.xlabel('t',fontsize=25)
plt.tick_params('x', labelbottom=False)
plt.yticks(fontsize=12)
ax2.xaxis.grid(linewidth=2)
ax2.yaxis.grid(linewidth=1)
ax2.legend(['data','MA 5'],fontsize=20)
plt.grid(True)
plt.savefig('Convolution_MA.png')
plt.show()
