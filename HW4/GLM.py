# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 17:25:26 2022

@author: waseem
"""


from __future__ import print_function
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
# from statsmodels.compat import urlopen
import statsmodels.formula.api as smf
from statsmodels.graphics.api import interaction_plot, abline_plot
from statsmodels.stats.anova import anova_lm
np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.width", 100)


# OLS related code
df = pd.read_excel('HW4/Datasets/us_macro_quarterly.xlsx')

xdata=df.loc[:,['JAPAN_IP', 'GS10', 'GS1', 'TB3MS', 'UNRATE', 'EXUSUK']].values
ydata=df.loc[:,'PCECTPI'].values
xdata = sm.add_constant(xdata) # adding a constant

# Fit all the data
model = sm.OLS(ydata, xdata).fit()
predictions = model.predict(xdata) 

print_model = model.summary()
print(print_model)
# with open('us_macro_quarterly.txt', 'w') as fh:
#     fh.write(model.summary().as_text())


# Use one of the train test split methods
# Train test split when order is not important
X_train, X_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.20,random_state=42)

# Train test split with 80/20 with last 20% data as the test when order is needed
X_train = xdata[0:176]; X_test = xdata[176:]
Y_train = ydata[0:176]; Y_test = ydata[176:]


model = sm.OLS(y_train,X_train).fit()

#In sample predictions
in_sample_predictions = model.predict(X_train) 

#Out of sample predictions
out_sample_predictions = model.predict(X_test)

fig = plt.figure(figsize=(150,70))
ax = plt.subplot()
ax.plot(df.loc[0:len(y_train)-1,'Date'],y_train,'k',linewidth=2.5)
ax.plot(df.loc[0:len(y_train)-1,'Date'],in_sample_predictions,'b',linewidth=2.5)
ax.plot(df.loc[len(y_train):,'Date'],y_test,'k--',linewidth=2.5)
ax.plot(df.loc[len(y_train):,'Date'],out_sample_predictions,'b--',linewidth=2.5)
ax.set_xlabel('Time');ax.set_ylabel('PCECTPI'); 
ax.legend(['training','in sample predictions','test','out sample predictions'],fontsize=100)
plt.show()


# Manually computing F-statistic
# Comparing with the intercept only model based on Null Hypothesis
print(anova_lm(sm.OLS(ydata, np.ones(len(ydata))*ydata.mean()).fit(),sm.OLS(ydata,xdata).fit()))

# Comparing with a less complex model
print(anova_lm(sm.OLS(ydata, df.loc[:,['JAPAN_IP', 'UNRATE', 'EXUSUK']]).fit(),sm.OLS(ydata,xdata).fit(), typ=1))

# Comparing with itself
print(anova_lm(sm.OLS(ydata, xdata).fit(),sm.OLS(ydata,xdata).fit(), typ=1))

# OLS related code block ends here



# GLM using Poisson Regression
# Smokers Age Poisson Regression discussed in the class
ds = pd.read_excel('HW4/Datasets/Smokers_Age.xlsx')
ds['PersonYears'] = np.log(ds['PersonYears'])
X = ds.loc[:,['Agecat','Smoke','Agecatsq','Smokeage','PersonYears']].values.tolist()
y = ds.Deaths.values.tolist()
# poisson_results = sm.GLM(y, X, family=sm.families.Poisson()).fit()
poisson_results = smf.glm('Deaths ~ Agecat + Smoke + Agecatsq + Smokeage + PersonYears', data = ds, family=sm.families.Poisson()).fit()

# y_avg = poisson_results.predict(X)
y_avg = poisson_results.predict()

fig = plt.figure(figsize=(20, 14))
ax1 = plt.subplot(211)
plt.plot(y_avg[0:5],'r-',linewidth=3.5)
plt.plot(y[0:5],'k',linewidth=3.5)
plt.ylabel('Smoker Deaths',fontsize=25)
plt.tick_params('x', labelbottom=False)
plt.yticks(fontsize=12)
ax1.xaxis.grid(linewidth=2)
ax1.yaxis.grid(linewidth=2)
ax1.legend(['Expected deaths','Observations'],fontsize=20)
plt.grid(True)

ax2 = plt.subplot(212)
plt.plot(y_avg[5:10],'r-',linewidth=3.5)
plt.plot(y[5:10],'k',linewidth=3.5)
plt.ylabel('Non Smokers Deaths',fontsize=25)
plt.xlabel('Age Category',fontsize=25)
plt.tick_params('x', labelbottom=True)
plt.yticks(fontsize=12)
ax2.xaxis.grid(linewidth=2)
ax2.yaxis.grid(linewidth=1)
ax2.legend(['Expected deaths','Observations'],fontsize=20)
# plt.savefig('Smokers_Poisson_Regression.png')
plt.show()

# with open('Smokers_Regression_Summary.txt', 'w') as fh:
#     fh.write(poisson_results.summary().as_text())




from scipy.stats import t

def t_score_probability(t_score, degrees_of_freedom):
  """
  Calculates the probability (p-value) for a given t-score.

  Args:
    t_score: The calculated t-score.
    degrees_of_freedom: The degrees of freedom for the t-distribution.

  Returns:
    The p-value associated with the t-score.
  """
  p_value = t.sf(np.abs(t_score), degrees_of_freedom) * 2
  return p_value

