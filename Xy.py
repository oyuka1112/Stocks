import pandas as pd
import numpy as np

#region Import data

def get_prices_df(percent_change=False, path='data/stock_prices.csv'):
    prices = pd.read_csv(path).set_index('date')
    prices.index = pd.to_datetime(prices.index)
    if percent_change:
        prices = prices.pct_change()
    return prices

def get_volumes_df(path='data/stock_volumes.csv'):
    vols = pd.read_csv(path).set_index('date')
    vols.index = pd.to_datetime(vols.index)
    vols = vols.fillna(0)
    return vols

def get_info_df(path='data/stock_info.csv'):
    info = pd.read_csv(path).set_index('Instrument')
    return info

def get_listings_df(path='data/listings.csv'):
    listings = pd.read_csv(path).set_index('date')
    listings.index = pd.to_datetime(listings.index)
    return listings

def get_data_defaults():
    return get_prices_df(), get_volumes_df(), get_info_df(), get_listings_df()

#endregion

#region Market data

def get_market_Xy(target_id=None, target_ahead_by=0, percent_change=False, path='data/stock_prices.csv'):
    """
    Returns the features (X) and target matrices (y) as DataFrames.
    Result can be stored into two variables at once or into an iterable of size 2.
    
    Parameters
    ----------
    target_id : str
        Stock ID for the target matrix y. If not given, no target matrix will be set.
    target_ahead_by : int, default 1 (day)
        How many days ahead of the datetime index the target matrix will be set.
    percent_change : bool, default True
        Whether or not to return matrices containing percent changes (True) or prices (False).
    path : str, default 'data/stock_prices.csv'
        Filepath for the prices csv file.
        
    Returns
    -------
    X: pd.DataFrame
        Features matrix X. Rows are dates; columns are individual stock IDs.
    y: pd.DataFrame
        Target matrix y for stock_ID = target_ID. Rows are dates; column is labeled to match target_ahead_by.
        Returns None if no stock_id is given for parameter target_id.
    """
    X = get_prices_df(percent_change=percent_change, path=path)
    y = None
    
    if target_id is not None:
        y = X.loc[:, [target_id]].shift(-target_ahead_by)
        y.rename({target_id: '{} +{} day'.format(target_id, target_ahead_by)}, axis=1, inplace=True)

    return X, y

def get_y(target_id=None, target_ahead_by=0, path='data/stock_prices.csv'):
    y = pd.read_csv(path, usecols=["date", target_id]).set_index("date")
    y.index = pd.to_datetime(y.index)
    
    if target_ahead_by != 0:
        y.rename({target_id: '{} +{} day'.format(target_id, target_ahead_by)}, axis=1, inplace=True)
        
    return y.shift(-target_ahead_by)
    

def get_delayed_X(stock, period_start=1, period_stop=20, period_step=1, end_date = "2021-06-20", days_before = 120, percent_change=False, path='data/stock_prices.csv'):
    market_X, _ = get_market_Xy(percent_change=percent_change, path=path)
    X = pd.DataFrame(market_X.loc[:, stock])
    for day in range(period_start, period_stop, period_step):
        X[str(day)+ " day delay"] = X[stock].shift(periods=day)

    X = X.drop(stock, axis=1)
    X = X.loc[:end_date].tail(days_before)
    return X

#endregion