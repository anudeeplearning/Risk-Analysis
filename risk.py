# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 12:39:36 2020

@author: Andy
"""
#import the necessary packages
import pandas as pd
#import the data
returns=pd.read_csv("C:/Users/ABCD/Videos/IPCAWP/data/Portfolios_Formed_on_ME_monthly_EW.csv",header=0,index_col=0,parse_dates=True,na_values=-99.99 )

#returns.head()
#Here we are focussing on 1 month low and High. We calculate the returns and the risk involved in the stock
per_returns=returns['Lo 10']
per_returns=per_returns.dropna()
per_returns=per_returns.pct_change()
per_returns=per_returns.dropna()
per_returns
volatility=per_returns.std()
risk_return_ratio=per_returns.mean()/volatility
risk_return_ratio

#Drawdowns- worst possible loss which is from local peak to minima

rets=returns[['Lo 10','Hi 10']]
rets.columns=['SmallCap','LargeCap']
rets=rets/100
rets.plot.line()
#cleaning up the data
rets.index=pd.to_datetime(rets.index,format="%Y%m")                             #convert the index format to date
#rets.index=rets.index.to_period('M')                                                  #remove date as it doesn't have relevance
rets.head()
rets.info()

#Computing drawdowns
#compute a wealth index, compute previous peaks, compute drawdown
wealth_index=1000*(1+rets["LargeCap"]).cumprod()
wealth_index.head()
wealth_index.plot.line()
previous_peaks=wealth_index.cummax()
previous_peaks.plot.line()
drawdown=(wealth_index-previous_peaks)/previous_peaks
drawdown.plot()
drawdown["1975":].min()
drawdown["1975":].idxmin()

def drawdown(return_series:pd.Series):
    """
    Takes a time series of asset returns
    computes and returns a DataFrame that contains
    the wealth index
    the previous peaks
    percentage drawdowns
    """
    wealth_index=1000*(1+return_series).cumprod()
    previous_peaks=wealth_index.cummax()
    drawdown=(wealth_index-previous_peaks)/previous_peaks
    return pd.DataFrame({
            "Wealth":wealth_index,
            "Peaks" :previous_peaks,
            "Drawdown":drawdown
            })
    

drawdown(rets["LargeCap"]).head()
drawdown(rets["LargeCap"])[["Wealth","Peaks"]].head()
drawdown(rets["LargeCap"])[["Wealth","Peaks"]].plot()



