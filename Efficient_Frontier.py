# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 08:26:47 2020

@author: Andy
"""
import pandas as pd
import numpy as np
from IPCAM import edhec_risk_kit as erk
import ipywidgets as widgets
from IPython.display import display

ind=erk.get_ind_returns()
erk.drawdown(ind["Food"])["Drawdown"].plot.line()

cols_of_interest=["Food","Smoke","Coal","Beer","Fin"]
erk.var_gaussian(ind[cols_of_interest],modified=True)
erk.var_gaussian(ind,modified=True).sort_values().tail()
erk.var_gaussian(ind,modified=True).sort_values().head()
erk.var_gaussian(ind,modified=True).sort_values().plot.bar()

erk.sharpe_ratio(ind["2000":],0.03,12).sort_values().plot.bar(title="Industry Sharpe Ratios 1926-2018",figsize=(12,5),color="goldenrod")

er=erk.annualize_rets(ind["1996":"2000"],12)
er.sort_values().plot.bar()
cov=ind["1996":"2000"].cov()

def portfolio_return(weights,returns):
    """
    Weights -> Returns
    """
    return weights.T @ returns

def portfolio_vol(weights,covmat):
    """
    Weights -> Vol
    """
    return (weights.T@ covmat@ weights)**0.5

l=["Food","Beer","Smoke","Coal"]
er[l]
cov.loc[l,l]

weights=np.repeat(1/4,4)
erk.portfolio_return(weights,er[l])
erk.portfolio_vol(weights,cov.loc[l,l])

###2 Asset Frontier
l=["Games","Fin"]
n_points=20
weights=[np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
rets=[erk.portfolio_return(w,er[l]) for w in weights]
vols=[erk.portfolio_vol(w,cov.loc[l,l]) for w in weights]
ef=pd.DataFrame({"R":rets,"Vol":vols})
ef.plot.scatter(x="Vol",y="R")

erk.plot_ef2(25,er[l],cov.loc[l,l])

w15= erk.minimize_vol(0.15,er[l],cov.loc[l,l])
vol15=erk.portfolio_vol(w15,cov.loc[l,l])
vol15


l=["Smoke","Fin","Games","Coal"]
erk.plot_ef(25,er[l],cov.loc[l,l])

ax=erk.plot_ef(20,er,cov)
ax.set_xlim(left=0.0)
rf=0.1
w_msr=erk.msr(rf,er,cov)
r_msr=erk.portfolio_return(w_msr,er)
vol_msr=erk.portfolio_vol(w_msr,cov)
# Add CML
cml_x=[0,vol_msr]
cml_y=[rf,r_msr]
ax.plot(cml_x,cml_y,color="green",marker="o",linestyle="dashed")
ax.figure

erk.plot_ef(20,er,cov,show_cml=True,riskfree_rate=0.1)


ind_return=erk.get_ind_returns()
ind_size=erk.get_ind_size()
ind_nfirms=erk.get_ind_nfirms()
ind_mktcap=ind_nfirms*ind_size
total_mktcap=ind_mktcap.sum(axis="columns")
total_mktcap.plot()
ind_capweight=ind_mktcap.divide(total_mktcap,axis="rows")
ind_capweight[["Fin","Steel"]].plot()
ind_capweight["2018-12"].T.sort_values(by=['2018-12']).plot.bar(figsize=(20,12))

total_market_return=(ind_capweight*ind_return).sum(axis="columns")
total_market_return.plot()
total_market_index=erk.drawdown(total_market_return).Wealth
total_market_index.plot(title="Total Market CapWeighted Index 1926-2018")
ax=total_market_index["1980":].plot()
ax.plot(total_market_index["1980":].rolling(window=36).mean())
ax.figure
tmi_tr36rets=total_market_return.rolling(window=36).aggregate(erk.annualize_rets,periods_per_year=12)
ay = tmi_tr36rets.plot(label='Trailing 36 month return',secoondary_y=True)
ay.plot(total_market_return,legend=True)
ay.figure

#Rolling Correlation along with multi-index and .groupby
ts_corr = ind_return.rolling(window=36).corr().dropna()
ts_corr.tail()
ts_corr.index.names=['date','industry']
ts_tr36corr=ts_corr.groupby(level='date').apply(lambda cormat:cormat.values.mean())
ts_tr36corr.plot()

ay = tmi_tr36rets.plot(label='Trailing 36 month return')(secondary_y=True)
ay.plot(ts_tr36corr,label='Trailing 36 month correlation')
ay.legend()
ay.figure

ay=tmi_tr36rets.plot(label='Trailing 36 month return')
ay.secondary_y()
az=ts_tr36corr.plot(label='Trailing 36 month correlation',secondary_y=True)
ay.plot(ts_tr36corr,label='Trailing 36 month correlation')




ind_return=erk.get_ind_returns()
tmi_return=erk.get_total_market_index_returns()
risky_r = ind_return["2000":][["Steel","Fin","Beer"]]

safe_r=pd.DataFrame().reindex_like(risky_r)
safe_r[:]=0.03/12
start=1000
floor=.8

"""
Cushion : Asset Value-Floor Value
Compute an allocation to safe and risky assets  --> m*risk budget
Recompute the asset value bassed on the returns
"""

def compound1(r):
    return (1+r).prod()-1

###compound2 is very efficient. avoid for loop as it takes a lot of time
def compound2(r):
    return np.expm1(np.log1p(r).sum())


dates=risky_r.index
n_steps=len(dates)
account_value=start
floor_value=start*floor
m=3
account_history=pd.DataFrame().reindex_like(risky_r)
cushion_history=pd.DataFrame().reindex_like(risky_r)
risky_w_history=pd.DataFrame().reindex_like(risky_r)

for step in range(n_steps):
    cushion = (account_value-floor_value)/account_value
    risky_w = m*cushion
    risky_w=np.minimum(risky_w,1)
    risky_w=np.maximum(risky_w,0)
    safe_w=1-risky_w
    risky_alloc=account_value*risky_w
    safe_alloc=account_value*safe_w
    #updating accounting value for this time step
    account_value=risky_alloc*(1+risky_r.iloc[step])+safe_alloc*(1+safe_r.iloc[step])
    #save values in history
    cushion_history.iloc[step]=cushion
    risky_w_history.iloc[step]=risky_w
    account_history.iloc[step]=account_value
    
risky_wealth=start+(1+risky_r).cumprod()

btr=erk.run_cppi(tmi_return["2007":],drawdown=0.25)
 
ax=btr["Wealth"].plot(figsize=(12,6),label="Wealth",legend=True)
btr["Risky Wealth"].plot(ax=ax,style="k--",label="Risky Wealth",legend=True)
ax.legend()
ax.figure

p = erk.gbm(n_years=10,n_scenarios=3)
p.plot()

gbm_controls = widgets.interactive(erk.show_gbm, n_scenarios=(1,20,5),mu=(0,0.2,0.01),sigma=(0,.3,.01) )
display(gbm_controls)

liabilities=pd.Series(data=[1,1.5,2,2.5],index=[3,3.5,4,4.5])
erk.pv(liabilities,0.03)

erk.funding_ratio(5,liabilities,0.03)
erk.cir(n_scenarios=10).plot(legend=False)


a_0 = 0.75
rates, bond_prices = erk.cir(n_years=10, r_0=0.03, b=0.03, n_scenarios=10)
liabilities=bond_prices
zcbond_10 = pd.Series(data=[1], index=[10])
zc_0 = erk.pv(zcbond_10, r=0.03)
n_bonds = a_0/zc_0
av_zc_bonds=n_bonds*bond_prices
av_cash= a_0*(rates/12+1).cumprod()
av_cash.plot(legend=False)

av_zc_bonds.plot(legend=False)
(av_cash/liabilities).pct_change().plot(title="Returns of funding Ratio with Cash (10 scenarios)", legend=False)
(av_zc_bonds/liabilities).pct_change().plot(title="Returns of funding Ratio with Cash (10 scenarios)", legend=False)


a_0=0.75
rates, bond_prices = erk.cir(n_years=10, r_0=0.03, b=0.03, n_scenarios=1000)
liabilities=bond_prices
zc_0 = erk.pv(zcbond_10, r=0.03)
n_bonds = a_0/zc_0
av_zc_bonds=n_bonds*bond_prices
av_cash= a_0*(rates/12+1).cumprod()
tfr_cash = av_cash.iloc[-1]/liabilities.iloc[-1]
tfr_zc_bonds = av_zc_bonds.iloc[-1]/liabilities.iloc[-1]
tfr_cash.plot.hist(label="Cash",bins=100,legend=True)
tfr_zc_bonds.plot.hist(label="ZC Bonds", bins=100, legend=True, secondary_y=True)