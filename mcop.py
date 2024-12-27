import datetime as dt 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Getting the necessary data
def getData(stocks, start, end):
    stockData = yf.download(stocks, start, end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covarianceMatrix = returns.cov()
    return meanReturns, covarianceMatrix

stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO'] # Random stocks
stocks = [stock + '.AX' for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)
meanReturns, covarianceMatrix = getData(stocks, startDate, endDate)
print(meanReturns)


