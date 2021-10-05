import numpy as np
import seaborn as sns
import pandas as pd
import random
import datetime as dt
import matplotlib.pyplot as plt
import scipy.optimize as sco

def get_prev_next_trading_day(prices, date):
    date = pd.to_datetime(date)
    return prices[prices.index < date].index[-1], prices[prices.index > date].index[0]

def get_return(weights, portfolio_df_returns):
    return np.sum(portfolio_df_returns.mean() * weights) * len(portfolio_df_returns.index)

def get_volatility(weights, portfolio_df_returns):
    return np.sqrt(np.dot(weights.T, np.dot(portfolio_df_returns.cov(), weights))) * np.sqrt(len(portfolio_df_returns.index))
    
def connect_long_short(long_df, short_df):
    return long_df.join(short_df)

#region Weight Optimization

def monte_carlo_SR(portfolio_df_returns, num_times=1000, risk_free_rate=0):
    optimized_weights = np.zeros((num_times, len(portfolio_df_returns.columns)))
    sharpe_ratios = np.zeros(num_times)
    
    for i in range(num_times):
        weights = np.array(np.random.random(len(portfolio_df_returns.columns)))
        weights = weights/np.sum(weights)
        
        optimized_weights[i, :] = weights
        weighted_daily_returns = (portfolio_df_returns*weights).sum(axis=1)
        sharpe_ratios[i] = (weighted_daily_returns.mean()-risk_free_rate)/weighted_daily_returns.std()

    return optimized_weights[sharpe_ratios.argmax(), :]

def max_SR(portfolio_df_returns, risk_free_rate=0):
    num_assets = len(portfolio_df_returns.mean())
    args = (portfolio_df_returns, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))

    def neg_sharpe_ratio(weights, portfolio_df_returns, risk_free_rate=0):
        vol, ret = get_volatility(weights, portfolio_df_returns), get_return(weights, portfolio_df_returns)
        return -(ret - risk_free_rate) / vol

    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def min_variance(portfolio_df):
    num_assets = len(portfolio_df.mean())
    args = (portfolio_df)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(get_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def test_all_models(portfolio_df, portfolio_df_actual, num_times_monte_carlo=1000, risk_free_rate=0):
    print('Monte Carlo returns:', np.sum(portfolio_df_actual.mean() * monte_carlo_SR(portfolio_df, num_times_monte_carlo)))
    print('Maximize Sharpe Ratio:', np.sum(portfolio_df_actual.mean() * max_SR(portfolio_df, risk_free_rate)['x']))
    print('Minimized Variance:', np.sum(portfolio_df_actual.mean() * min_variance(portfolio_df)['x']))

#endregion
