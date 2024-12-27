import datetime as dt 
import yfinance as yf
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# Getting the necessary data
def getData(stocks, start, end):
    stocksData = yf.download(stocks, start, end)['Close'] # Downloading stock closing prices
    returns = stocksData.pct_change() # Calculating daily returns with the pct_change() func
    meanReturns = returns.mean() # Average daily returns
    covarianceMatrix = returns.cov() # Measure of how stock returns move together. Captures relationships between stock returns.
    return meanReturns, covarianceMatrix

stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO'] # Random stocks
stocks = [stock + '.AX' for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=250)
meanReturns, covarianceMatrix = getData(stocks, startDate, endDate)

weights = np.random.random(len(meanReturns)) # Random weight generation to assign proportions to each stock in portfolio, representing the fraction of total investment allocated to each stock.
weights /= np.sum(weights) # Basically normalizing the weights to ensure their sum = 1
print(weights)

# Monte Carlo method. First, number of simulations:
mc_sim = 200 # Number of simulations
time_d = 200 # Time 
meanMatrix = np.full(shape=(time_d, len(weights)), fill_value=meanReturns).T # a matrix where each row contains the mean returns 
pf_sim = np.full(shape=(time_d, mc_sim), fill_value=0.0) # 'pf' = portfolio, a 2D array to store portfolio values for all simulations over time
pf_initialValue = 5000

for x in range(0, mc_sim):
    # I'll be using Cholesky Decomposition:
    # Decomposes the covariance matrix into a lower triangular matrix, l.
    # This matrix transforms the independent random normal values (z) into correlated random values.

    z = np.random.normal(size=(time_d, len(weights))) # Random normal values to simulate stochastic returns. Purpose: Introduces randomness into the simulation to mimic the unpredictable nature of stock market movements.
    l = np.linalg.cholesky(covarianceMatrix) # Cholesky Decomposition to generate correlated returns based on the covariance matrix. Purpose: Accounts for correlations between stocks when simulating returns.
    dailyReturns = meanMatrix + np.inner(l, z) # Combines mean matrix and decomposed matrix with random values to model stock behavior. Adds randomness while incorporating correlations (via l) to generate more realistic stock returns.
    pf_sim[:,x] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*pf_initialValue # Calculates cumulative product of daily returns scaled by weights, simulating portfolio growth

plt.plot(pf_sim)
plt.ylabel('Portfolio Value')
plt.xlabel('Days')
plt.title('Monte Carlo Sim')
plt.show()



