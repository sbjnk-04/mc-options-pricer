import datetime as dt 
import yfinance as yf
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# Getting the necessary data
def getData(stocks, start, end):
    stocksData = yf.download(stocks, start, end)
    stocksData = stocksData['Close']
    returns = stocksData.pct_change()
    meanReturns = returns.mean()
    covarianceMatrix = returns.cov()
    return meanReturns, covarianceMatrix

stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO'] # Random stocks
stocks = [stock + '.AX' for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)
meanReturns, covarianceMatrix = getData(stocks, startDate, endDate)

weights = np.random.random(len(meanReturns)) 
weights /= np.sum(weights)
print(weights)


