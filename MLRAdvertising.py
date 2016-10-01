# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 11:43:46 2014

@author: kurita
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
import statsmodels.formula.api as sm

Addata = pd.read_csv('Advertising.csv')

#print Addata[0:100]

result = sm.ols(formula="Sales ~ TV + Radio + Newspaper", data=Addata).fit()

print result.summary()

result = sm.ols(formula="Sales ~ TV", data=Addata).fit()

print result.summary()

result = sm.ols(formula="Sales ~ Radio", data=Addata).fit()

print result.summary()

result = sm.ols(formula="Sales ~ Newspaper", data=Addata).fit()

print result.summary()

result = sm.ols(formula="Sales ~ TV + Radio", data=Addata).fit()

print result.summary()

# sklearn linear_model
#
# comvert the pandas data frame to np.array

print '\n###### Fiting the linear regression\n'

data = np.asarray(Addata)

y = data[:,4]
X = data[:,1:4]

#print data
#print y
#print X

lr = linear_model.LinearRegression()
lr.fit(X, y)

print 'coeffients are ', lr.intercept_, lr.coef_

