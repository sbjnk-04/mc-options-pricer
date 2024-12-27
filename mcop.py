import datetime as dt 
import yfinance as yf
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# Getting the necessary data
def getData(stocks, start, end):
    stocksData = yf.download(stocks, start, end)['Close']
    returns = stocksData.pct_change()
    meanReturns = returns.mean()
    covarianceMatrix = returns.cov()
    return meanReturns, covarianceMatrix

stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO'] # Random stocks
stocks = [stock + '.AX' for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=250)
meanReturns, covarianceMatrix = getData(stocks, startDate, endDate)

weights = np.random.random(len(meanReturns)) 
weights /= np.sum(weights)
print(weights)

# Monte Carlo method. First, number of simulations:
mc_sim = 200 
time_d = 200
meanMatrix = np.full(shape=(time_d, len(weights)), fill_value=meanReturns).T
pf_sim = np.full(shape=(time_d, mc_sim), fill_value=0.0) # 'pf' = portfolio
pf_initialValue = 5000

for x in range(0, mc_sim):
    # I'll be using Cholesky Decomposition
    z = np.random.normal(size=(time_d, len(weights)))
    l = np.linalg.cholesky(covarianceMatrix)
    dailyReturns = meanMatrix + np.inner(l, z)
    pf_sim[:,x] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*pf_initialValue

plt.plot(pf_sim)
plt.ylabel('Portfolio Value')
plt.xlabel('Days')
plt.title('Monte Carlo Sim')
plt.show()



