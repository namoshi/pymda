# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:53:04 2014

@author: kurita
"""

import numpy as np
import statsmodels.api as sm
from scipy import stats
from matplotlib import pyplot as plt

data = sm.datasets.star98.load()

data.exog = sm.add_constant(data.exog, prepend=False)

# Logistic Regression
glm_binom = sm.GLM(data.endog, data.exog, family=sm.families.Binomial())
res = glm_binom.fit()

print res.summary()


nobs = res.nobs
y = data.endog[:, 0] / data.endog.sum(1)
yhat = res.mu

plt.figure();
plt.scatter(yhat, y);
line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=False)).fit().params
fit = lambda x: line_fit[1] + line_fit[0] * x  # better way in scipy?
plt.plot(np.linspace(0, 1, nobs), fit(np.linspace(0, 1, nobs)));
plt.title('Model Fit Plot');
plt.ylabel('Observed values');
plt.xlabel('Fitted values');

plt.figure();
plt.scatter(yhat, res.resid_pearson);
plt.plot([0.0, 1.0], [0.0, 0.0], 'k-');
plt.title('Residual Dependence Plot');
plt.ylabel('Pearson Residuals');
plt.xlabel('Fitted values');

plt.figure();
resid = res.resid_deviance.copy()
resid_std = (resid - resid.mean()) / resid.std()
plt.hist(resid_std, bins=25);
plt.title('Histogram of standardized deviance residuals');

from statsmodels import graphics
graphics.gofplots.qqplot(resid, line='r');

plt.show()

