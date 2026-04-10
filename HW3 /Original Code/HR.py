#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 14:27:50 2026

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
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
import scipy
from scipy import linalg
import datetime


# cd /Users/waseem/Desktop/Waseem/Teaching/Forecasting/Code/Time\ Series\ Forecasting/HR

dv = pd.read_csv('sensor_hrv.csv')

ds = dv[['deviceId', 'ts_start', 'ts_end', 'HR']].copy()

# Create datetime object (UTC)
ds['ts_utc'] = pd.to_datetime(ds['ts_start'], unit='ms', utc=True)

ds = ds.sort_values(by=['deviceId', 'ts_start'])



def cosinor_model(t, mesor, amplitude, phase):
    return mesor + amplitude * np.cos(2 * np.pi * (t - phase) / 24)

# for x in np.unique(ds.deviceId):
for x in ['ab60','gd81','ub12','mh40']:

    fig = plt.figure(figsize=(50, 14))
    
    
    # Rotate x-axis tick labels by 45 degrees
    subset = ds[ds.deviceId == x]#.iloc[0:100]
    
    # 1. Prepare time 't' as hours since the start of data
    subset['hour_index'] = (subset['ts_utc'] - subset['ts_utc'].min()).dt.total_seconds() / 3600
    
    # 2. Fit the model
    params, _ = curve_fit(cosinor_model, subset['hour_index'], subset['HR'], 
                          p0=[subset['HR'].mean(), 10, 12])
    
    subset['circadian_cosinor'] = cosinor_model(subset['hour_index'], *params)


    plt.plot(subset['ts_utc'],subset['HR'])
    plt.plot(subset['ts_utc'],subset['HR'].rolling(window=60).mean(),linewidth=2)
    plt.plot(subset['ts_utc'],subset['circadian_cosinor'],linewidth=2,color='r')
    plt.ylabel('HR')
    plt.xlabel('Time t',labelpad=100,fontsize=7)

    # 1. Floor to the hour (keeps the date attached)
    # 2. Get unique values
    # 3. Sort them chronologically
    # unique_hourly_timestamps = sorted(subset['ts_utc'].dt.floor('h').unique())
    unique_hourly_timestamps = subset['ts_utc'].dt.floor('h').unique()
    
    # 4. Format as strings (e.g., "04-01 07:00")
    hourly_labels = [ts.strftime('%m-%d %H:00') for ts in unique_hourly_timestamps[::2]]
    
    print(hourly_labels[:5]) 
    # Output: ['04-01 07:00', '04-01 08:00', '04-01 09:00', ...]

    # 3. Set the ticks: Positions first, then the formatted Labels
    plt.xticks(
        # ticks=subset['ts_utc'], 
        # labels=[x.strftime('%H:%M') for x in subset['ts_utc']], 
        ticks = unique_hourly_timestamps[::2], labels = hourly_labels,
        fontsize=7,rotation=90
    )
    
    plt.legend(['HR','Circadian Rhythm','Circadian Rhythm'])
    plt.grid(axis='y')
    plt.savefig('Circadian_Rhythm_' + x +'.png')
    plt.show()

