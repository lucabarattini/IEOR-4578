# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 13:43:04 2022

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


dv = pd.read_excel('Viscosity.xlsx')
dw = pd.read_excel('WholeFood.xlsx')
dp = pd.read_excel('Pharmaceutical.xlsx')

# Butterworth LP Filter
# Fred Ice Cream Frozen Deserts Production Index
dicfn = pd.read_csv('IPN31152N_Fred_Industrial Production Manufacturing Non-Durable Goods Ice Cream and Frozen Dessert.csv')
dicfn.DATE = pd.to_datetime(dicfn.DATE)
t = [i%365 for i in range(0, len(dicfn.DATE[dicfn.DATE.dt.year > 2015]))]
sig = dicfn.IPN31152N[dicfn.DATE.dt.year > 2015].values
RMSE = 10000000000
fss=0
order=0

for i in np.arange(3,20):
    for j in np.arange(i,30):
        sos = signal.butter(i, 1, 'lowpass', fs=j, output='sos')
        filtered = signal.sosfilt(sos, sig)
        freqs, psd = signal.welch(sig)
        freqs, psd = signal.welch(sig-filtered)
        MSE=100000000000000000000
        MSE = np.square(sig-filtered).mean() 
        if(RMSE > math.sqrt(MSE)):
            RMSE = math.sqrt(MSE)
            order = i
            fss = j
            fig = plt.figure(figsize=(20, 14))
            ax1 = plt.subplot(511)
            plt.plot(t, sig)
            ax2 = plt.subplot(512)
            plt.plot(freqs, psd)
            ax3 = plt.subplot(513)
            plt.plot(t, filtered)
            ax4 = plt.subplot(514)
            plt.plot(t,sig-filtered)
            plt.ylim(-25,25)
            ax5 = plt.subplot(515)
            plt.plot(freqs, psd)
            plt.show()
            
sos = signal.butter(order, 1, 'lowpass', fs=fss, output='sos')
filtered = signal.sosfilt(sos, sig)
fig = plt.figure(figsize=(30, 24))
ax1 = plt.subplot(511)
plt.plot(t, sig,'k')
plt.xlabel('t',fontsize=25)
plt.legend(['Production Index'],fontsize=20)
freqs, psd = signal.welch(sig)
plt.grid(True)
ax2 = plt.subplot(512)
plt.plot(freqs, psd,'k')
plt.xlabel('freq',fontsize=25)
plt.legend(['PSD of Production Index'],fontsize=20)
plt.grid(True)
ax3 = plt.subplot(513)
plt.plot(t, filtered,'k')
plt.xlabel('t',fontsize=25)
plt.legend(['Production Index LP'],fontsize=20)
plt.grid(True)
ax4 = plt.subplot(514)
plt.plot(t,sig-filtered,'k')
plt.ylim(-25,25)
plt.xlabel('t',fontsize=25)
plt.legend(['Production Index - LP'],fontsize=20)
freqs, psd = signal.welch(sig-filtered)
plt.grid(True)
ax5 = plt.subplot(515)
plt.plot(freqs, psd,'k')
plt.xlabel('freq',fontsize=25)
plt.legend(['PSD of filtered Production Index'],fontsize=20)
plt.grid(True)
# plt.savefig('ProductionIndex_Butterworth.png')
plt.show()
MSE = np.square(sig-filtered).mean()
RMSE = math.sqrt(MSE)
