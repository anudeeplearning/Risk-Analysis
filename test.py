# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:05:12 2020

@author: Andy
"""

from IPCAM import edhec_risk_kit as erk

returns=erk.get_ffme_returns()
erk.drawdown(returns["SmallCap"])["Drawdown"].min()
erk.drawdown(returns['1975':]["SmallCap"])["Drawdown"].min()

hfi=erk.get_hfi_returns()
hfi.head()

pd.concat([hfi.mean(),hfi.median(),hfi.mean()>hfi.median()],axis="columns")
erk.skewness(hfi).sort_values()

import scipy.stats
scipy.stats.skew(hfi)

erk.kurtosis(hfi).sort_values()

scipy.stats.jarque_bera(hfi)

erk.is_normal(hfi)
#the above function is taking the whole matrix into consideration but we need for individual type of instrument
hfi.aggregate(erk.is_normal)

ffme=erk.get_ffme_returns()
erk.skewness(ffme)
erk.kurtosis(ffme)
ffme.aggregate(erk.is_normal)



hfi=erk.get_hfi_returns()
#Semideviation is deviation of returns lower than zero
erk.semideviation(hfi)

#VaR and CVaR
###Value at risk

#Historic VaR
erk.var_historic(hfi)

#Parametric VaR-Gaussian 
erk.var_gaussian(hfi)

#Modified Cornish-Fischer VaR
erk.var_gaussian(hfi,modified=True)

var_list=[erk.var_gaussian(hfi),erk.var_gaussian(hfi,modified=True),erk.var_historic(hfi)]
comparison=pd.concat(var_list,axis=1)
comparison.columns=["Gaussian","Cornish-Fischer","Historic"]
comparison.plot.bar(title="EDHEC Hedge Fund Indices: VaR") 

#Conditional Value at Risk CVaR
erk.cvar_historic(hfi)

"""%load_ext autorelaod
%autoreload 2
pd.to_datetime"""