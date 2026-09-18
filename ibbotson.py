import numpy as np
import pandas as pd
from statsmodels.api import OLS
import matplotlib.pyplot as plt
import scipy
from statsmodels.api import stats

def verification(data):
    print('Shapiro-Wilk p = ', scipy.stats.shapiro(data)[1])
    print('Jarque-Bera p = ', scipy.stats.jarque_bera(data)[1])
    print('ACF p-value for Ljung-Box test = ', stats.acorr_ljungbox(data, lags = [5, 10])['lb_pvalue'].values)
    print('Same for absolute values = ', stats.acorr_ljungbox(abs(data), lags = [5, 10])['lb_pvalue'].values)

# reading the data file
DF = pd.read_excel('ibbotson.xlsx')
rate = DF['Long'].values
vol = DF['Volatility'].values[3:]
trueRet = np.log(1 + DF['Returns'].values[1:] - 0.01 * rate[:-1])

# Regression only for Ibbotson data 1926-2015
Reg = OLS(trueRet[:-10], pd.DataFrame({'const' : 1, 'duration': np.diff(rate)}).iloc[:-10]).fit()

print('Ibbotson data 1926-2015')
print(Reg.summary())
verification(Reg.resid)
print('stderr = ', np.std(Reg.resid))

# And now divide residuals by volatility
print('Residuals after Division by Volatility')
verification(Reg.resid[2:]/vol[:-10])