# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:12:18 2020

@author: Andy
"""
import pandas as pd
import numpy as np
import math
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize
import ipywidgets as widgets
from IPython.display import display

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
    
def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    me_m=pd.read_csv("C:/Users/ABCD/Videos/IPCAWP/data/Portfolios_Formed_on_ME_monthly_EW.csv",header=0,index_col=0,parse_dates=True,na_values=-99.99 )
    rets=me_m[['Lo 10','Hi 10']]
    rets.columns=['SmallCap','LargeCap']
    rets=rets/100
    rets.index=pd.to_datetime(rets.index,format="%Y%m")
    return rets

def get_ind_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    df=pd.read_csv("C:/Users/ABCD/Videos/IPCAWP/data/ind30_m_vw_rets.csv",header=0,index_col=0,parse_dates=True)/100
    df.index=pd.to_datetime(df.index,format="%Y%m").to_period("M")
    df.columns=df.columns.str.strip()
    return df

def get_ind_size():
    """
    Load the Fama-French Dataset for the marketcap of industries
    """
    df=pd.read_csv("C:/Users/ABCD/Videos/IPCAWP/data/ind30_m_size.csv",header=0,index_col=0,parse_dates=True)
    df.index=pd.to_datetime(df.index,format="%Y%m").to_period("M")
    df.columns=df.columns.str.strip()
    return df

def get_ind_nfirms():
    """
    Load the Fama-French Dataset for the number of firms in industries
    """
    df=pd.read_csv("C:/Users/ABCD/Videos/IPCAWP/data/ind30_m_nfirms.csv",header=0,index_col=0,parse_dates=True)
    df.index=pd.to_datetime(df.index,format="%Y%m").to_period("M")
    df.columns=df.columns.str.strip()
    return df

def get_total_market_index_returns():
    ind_return=get_ind_returns()
    ind_size=get_ind_size()
    ind_nfirms=get_ind_nfirms()
    ind_mktcap=ind_nfirms*ind_size
    total_mktcap=ind_mktcap.sum(axis="columns")
    ind_capweight=ind_mktcap.divide(total_mktcap,axis="rows")
    total_market_return=(ind_capweight*ind_return).sum(axis="columns")
    return total_market_return

def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi=pd.read_csv("C:/Users/ABCD/Videos/IPCAWP/data/edhec-hedgefundindices.csv",header=0,index_col=0,parse_dates=True,na_values=-99.99 )
    hfi=hfi/100
    hfi.index=hfi.index.to_period('M')
    return hfi

def skewness(r):
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    demeaned_r=r-r.mean()
    #use the population standard deviation, so set dof=0
    sigma_r=r.std(ddof=0)
    exp=(demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    demeaned_r=r-r.mean()
    #use the population standard deviation, so set dof=0
    sigma_r=r.std(ddof=0)
    exp=(demeaned_r**4).mean()
    return exp/sigma_r**4

def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be   a series or a dataframe
    """
    is_negative = r<0
    return r[is_negative].std(ddof=0)

def is_normal(r,level=0.01):
    """
    Applies Jarque-Bera test to determine if a series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepceted, False otherwise
    """
    #statistic, p_value = scipy.stats.jarque_bera
    result=scipy.stats.jarque_bera(r)
    return result[1] > level


def var_historic(r,level=5):
    """
    VaR Historic
    """
    if isinstance(r,pd.DataFrame):
        return r.aggregate(var_historic,level=level)
    elif isinstance(r,pd.Series):
        return -np.percentile(r,level)
    else:
        raise TypeError("Expected to be Series or DataFrame")
        
        
        
def var_gaussian(r,level=5,modified=False):
    """
    Returns the Parametric Gaussian VaR of a series or dataframe
    If "modified" is True then the modified VaR is returned using 
    the Cornish-Fisher modification
    """
    #compute the Z score assuming it was Gaussian using numpy
    z=norm.ppf(level/100)
    if modified:
        #modify the Z score bassed on observed skewness and kurtosis
        s=skewness(r)
        k=kurtosis(r)
        z=(z + (z**2-1)*s/6+ (z**3-3*z)*(k-3)/24 - (2*z**3-5*z)*(s**2)/36)
        
    return -(r.mean()+z*r.std(ddof=0))


def cvar_historic(r,level=5):
    """ 
    Computes the conditional VaR of a series or dataframe
    """
    if isinstance(r,pd.Series):
        is_beyond = r<= -var_historic(r,level=level)
        return -r[is_beyond].mean()
    elif isinstance(r,pd.DataFrame):
        return r.aggregate(cvar_historic(r,level=level))
    else:
        raise TypeError("Expected to be Series or DataFrame")
        
        
def sharpe_ratio(r,riskfree_rate,periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r-rf_per_period
    ann_ex_ret = annualize_rets(excess_ret,periods_per_year)
    ann_vol = annualize_vol(r,periods_per_year)
    return ann_ex_ret/ann_vol
    

def annualize_vol(r,periods_per_year):
    """
    Annualizes the vol of a set of returns
    """
    return r.std()*(periods_per_year**0.5)

def annualize_rets(r,periods_per_year):
    """
    Annualizes a set of returns
    """
    compounded_growth=(1+r).prod()
    n_periods=r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1
    

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

def plot_ef2 (n_points,er,cov,style=".-"):
    """
    Plots the 2-aset efficient frontier
    """
    if er.shape[0]!=2 or cov.shape[0]!=2:
        raise ValueError("Plot_ef2 can only plot 2-asset frontiers")
    weights=[np.array([w,1-w]) for w in np.linspace(0, 1, n_points)]
    rets=[portfolio_return(w,er) for w in weights]
    vols=[portfolio_vol(w,cov) for w in weights]
    ef=pd.DataFrame({"Returns":rets,"Volatility":vols})
    return ef.plot.line(x="Volatility",y="Returns",style=style)


def target_is_met(w,er):
    return target_return-portfolio_return(w,er)

def minimize_vol (target_return,er,cov):
    """
    target_ret -> W
    """
    n=er.shape[0]
    init_guess=np.repeat(1/n,n)
    bounds=((0.0,1.0),)*n 
    return_is_target={'type': 'eq','args':(er,),'fun':lambda weights, er:target_return-portfolio_return(weights,er)}
    weights_sum_to_1={'type':'eq','fun':lambda weights:np.sum(weights)-1}
    results=minimize(portfolio_vol,init_guess,args=(cov,),method='SLSQP',options={'disp':False},constraints=(return_is_target,weights_sum_to_1),bounds=bounds)
    return results.x


def optimal_weights(n_points,er,cov):
    """
    -> list of weights to run the optimiser on to minimise the vol
    """
    target_rs=np.linspace(er.min(),er.max(),n_points)
    weights=[minimize_vol(target_return,er,cov) for target_return in target_rs]
    return weights

def gmw(cov):
    """
    Returns the weight of the global minimum volatility portfolio
    given the covariance matrix
    """
    n=cov.shape[0]
    return msr(0,np.repeat(1,n),cov)

def plot_ef (n_points,er,cov,style=".-",show_cml=False,riskfree_rate=0,figsize=(20,10),show_ew=False,show_gmw=False):
    """
    Plots the N-aset efficient frontier and 
    """
    weights=optimal_weights(n_points,er,cov)
    rets=[portfolio_return(w,er) for w in weights]
    vols=[portfolio_vol(w,cov) for w in weights]
    ef=pd.DataFrame({"Returns":rets,"Volatility":vols})
    ax=ef.plot.line(x="Volatility",y="Returns",style=style,figsize=figsize)
    if show_ew:
        n=er.shape[0]
        w_ew=np.repeat(1/n,n)
        r_ew=portfolio_return(w_ew,er)
        vol_ew=portfolio_vol(w_ew,cov)
        #Display EW
        ax.plot([vol_ew],[r_ew],color= "goldenrod",marker="o",markersize=10)
    if show_gmw:
        w_gmw=gmw(cov)
        r_gmw=portfolio_return(w_gmw,er)
        vol_gmw=portfolio_vol(w_gmw,cov)
        #Display EW
        ax.plot([vol_gmw],[r_gmw],color= "goldenrod",marker="o",markersize=10)
    if show_cml:
        ax.set_xlim(left=0.0)
        w_msr=msr(riskfree_rate,er,cov)
        r_msr=portfolio_return(w_msr,er)
        vol_msr=portfolio_vol(w_msr,cov)
        # Add CML
        cml_x=[0,vol_msr]
        cml_y=[riskfree_rate,r_msr]
        ax.plot(cml_x,cml_y,color="green",marker="o",linestyle="dashed",markersize=12,linewidth=2)
    return ax.figure


def msr (riskfree_rate,er,cov):
    """
    Returns weights of the portfoliio that gives the maximum Sharpe ratio
    given the riskfree rate, expected returns and covariance matrix
    """
    n=er.shape[0]
    init_guess=np.repeat(1/n,n)
    bounds=((0.0,1.0),)*n 
    def neg_sharpe_ratio(weights,riskfree_rate,er,cov):
        """
        Returns the negative of Sharpe Ratio, given weights
        """
        r=portfolio_return(weights,er)
        vol=portfolio_vol(weights,cov)
        return -(r-riskfree_rate)/vol
    
    weights_sum_to_1={'type':'eq','fun':lambda weights:np.sum(weights)-1}
    results=minimize(neg_sharpe_ratio,init_guess,args=(riskfree_rate,er,cov,),method='SLSQP',options={'disp':False},constraints=(weights_sum_to_1),bounds=bounds)
    return results.x


def run_cppi(risky_r,safe_r=None,m=3, start=1000, floor=0.8, riskfree_rate=0.03,drawdown=None):
    """
    Run a backtest of the CPPI strategy, given aa set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    #set up CPPI parameters
    
    dates=risky_r.index
    n_steps=len(dates)
    account_value=start
    floor_value=start*floor
    peak=start
    
    if isinstance(risky_r,pd.Series):
        risky_r=pd.DataFrame(risky_r,columns=["R"])
    if safe_r is None:
        safe_r=pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:]=riskfree_rate/12
    #creating dataframes to make a history
    account_history=pd.DataFrame().reindex_like(risky_r)
    cushion_history=pd.DataFrame().reindex_like(risky_r)
    risky_w_history=pd.DataFrame().reindex_like(risky_r)
    
    for step in range(n_steps):
        if drawdown is not None:
            peak=np.maximum(peak,account_value)
            floor_value=peak*(1-drawdown)
        cushion = (account_value-floor_value)/account_value
        risky_w = m*cushion
        risky_w=np.minimum(risky_w,1)
        risky_w=np.maximum(risky_w,0)
        safe_w=1-risky_w
        risky_alloc=account_value*risky_w
        safe_alloc=account_value*safe_w
        #updating accountinn value for this time step
        account_value=risky_alloc*(1+risky_r.iloc[step])+safe_alloc*(1+safe_r.iloc[step])
        #save values in history
        cushion_history.iloc[step]=cushion
        risky_w_history.iloc[step]=risky_w
        account_history.iloc[step]=account_value
        
    risky_wealth=start*(1+risky_r).cumprod()
    backtest_result={"Wealth": account_history, "Risky Wealth":risky_wealth, "Risk Budget": cushion_history, "Risky Allocation": risky_w_history, "m":m, "start":start,"floor":floor, "risky_r":risky_r,"safe_r":safe_r}
    
    return backtest_result

def summary_stats(r,riskfree_rate=0.03):
    """
    Returns a DataFrame that contains aggregated summary for the returns in columns of r
    """
    ann_r=r.aggregate(annualize_rets,periods_per_year=12)
    ann_vol=r.aggregate(annualize_vol,periods_per_year=12)
    ann_sr=r.aggregate(sharpe_ratio,riskfree_rate=riskfree_rate,periods_per_year=12)
    dd=r.aggregate(lambda r:drawdown(r).Drawdown.min())
    skew=r.aggregate(skewness)
    kurt=r.aggregate(kurtosis)
    cf_var5=r.aggregate(var_gaussian,modified=True)
    hist_cvar5=r.aggregate(cvar_historic)
    return pd.DataFrame({"Annualized Return":ann_r,"Annualized Vol":ann_vol,"Skewness":skew,"Kurtosis":kurt,"Corn-Fischer VaR":cf_var5,"Historic CVaR(5%)":hist_cvar5,"Sharpe Ratio":ann_sr,"Max Drawdown":dd})


def gbm(n_years=10,n_scenarios=1000,mu=0.07,sigma=0.15,steps_per_year=12,s_0=100.0,prices=True):
    """
    Evolutionof a stock price using Geometric Brownian Motion Model
    """
    dt=1/steps_per_year
    n_steps=int(n_years*steps_per_year)+1
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt,scale=(sigma*np.sqrt(dt)),size=(n_steps,n_scenarios))
    #xi = np.random.normal(size=(n_steps,n_scenarios))
    rets_plus_1[0] = 1
    
    ret_val=s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    # to prices
    
    return ret_val

def show_gbm(n_scenarios,mu,sigma):
    """
    Draw the results of a stock price evolution under a Geometric Brownian Motion model
    """
    s_0=100
    prices = gbm(n_scenarios=n_scenarios,mu=mu,sigma=sigma,s_0=s_0)
    ax=prices.plot(legend=False,color="indianred",alpha =0.5, linewidth=2, figsize=(12,5))
    ax.axhline(y=s_0,ls=":",color="black")
    #draw a dot at origin
    ax.plot(0,s_0,color='darkred',alpha=0.2)
    
def discount(t,r):
    """
    Compute the price of a pure discount bond that pays a dollar at a time, given interest rate r
    """
    return (1+r)**(-t)

def pv(l,r):
    """
    Computes the present value of a sequence of liabilities
    l is indexed by the time, and the values are the amounts of each liability
    returns the present value of the sequence
    """
    dates=l.index
    discounts = discount(dates,r)
    return (discounts*l).sum()

def funding_ratio(assets,liabilities,r):
    """
    Computes the funding ratio of some assets given liabilites and interest rates
    """
    return assets/pv(liabilities,r)

def inst_to_ann(r):
    """
    Converts short rate to annualized rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    Converts annualized rate to short rate
    """
    return np.log1p(r)

def cir(n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12,r_0=None):
    """
    drt=a*(b-rt)*dt+sigma*sqrt(rt)*(dWt)
    Implements the CIR model for interest rate
    """
    if r_0 is None:
        r_0=b
    r_0=ann_to_inst(r_0)
    dt=1/steps_per_year
    
    num_steps = int(n_years*steps_per_year)
    shock=np.random.normal(0,scale=np.sqrt(dt),size=(num_steps,n_scenarios))
    rates=np.empty_like(shock)
    rates[0]=r_0
    
    #For price Generation
    h=math.sqrt(a**2+2*sigma**2)
    prices=np.empty_like(shock)
    ####
    
    def price(ttm,r):
        _A=((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B=(2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P=_A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    #####
    
    for step in range(1,num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t) + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well...
        prices[step] = price(n_years-step*dt, rates[step])
        
    rates=  pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    #for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    
    return rates,prices
    