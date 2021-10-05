#TODO If start_date or end_date are not trading days, throw an error

import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import time
import math

from diff_cap_packages import Xy

prices, volumes, info, _ = Xy.get_data_defaults()
listings = Xy.get_listings_df('data/listings_july2021.csv')

def get_dataframe_prices(end_date=prices.index[-1], num_days=200, no_NaN_threshold=200, volume_threshold=150):
    """
    This returns our investible universe

    start_date          : datetime object
    end_date            : datetime object
    no_NaN_days         : int
    volume_threshold    : int

    The investible universe is start_date (inclusive) and it ends at end_date (inclusive)
    Default dates are first day of the df and last day
    """

    end_date = pd.to_datetime(end_date)
    start_date = prices[:end_date][:-num_days].index[-1]

    data = prices[(prices.index >= start_date) & (prices.index <= end_date)]        # Stock prices from start_date to end_date
    data = wipe_dead(data)                                                          # Stock prices of ONLY active stocks
    data = wipe_not_listed(data)
    data = wipe_nodata(data, no_NaN_threshold)                                      # Stock prices of stocks with data for most recent NaN_days
    data = wipe_low_volumes(data, volume_threshold)                                 # Stock prices of stocks with mean volume >= volume_threshold
    return data

def wipe_not_listed(X):
    return X[[i for i in X.columns if i in listings.stock.values]]

def wipe_dead(X):
    """
    This wipes the stock if it is not currently active
    """
    return X.reindex(columns = info[info['ESTAT'] == 'ACT.'].index.values)
    
def wipe_nodata(X, no_NaN_threshold=200):
    """
    This wipes the stock if there are NaN values for price in the last min_days amount
    """
    return X[X.tail(no_NaN_threshold - 1).dropna(axis=1).columns]

def wipe_low_volumes(X, threshold=150):
    """
    This wipes the stock if it's mean volume over the course of time is less than the threshold
    """
    return X.loc[:, (volumes[X.columns].mean() >= threshold)]

#     """
#     Returns the listings (P) as a DataFrames.

#     Parameters
#     ----------
#     start_date : str
#         Start date that the stock was listed.
#     end_date : str
        
#     data : DataFrame, default listings


#     Returns
#     -------
#     P: pd.DataFrame
       

#     """
#     k = datetime.strptime(start_date, "%d/%m/%Y") 
#     l = datetime.strptime(end_date, "%d/%m/%Y")

#     start_date = k.strftime("%d/%m/%Y")
#     end_date = l.strftime("%d/%m/%Y")

#     P = data.loc[start_date:end_date]
    
   
#     return P

#low_volumes will take the  

# def low_volumes(k, start_date, end_date):
#     volumes.index = pd.to_datetime(volumes.index)
 
    
#     data = volumes.iloc[start_date: end_date]
#     filled_volume = data.fillna(0)
#     mean_data = filled_volume.mean()
#     lists = []
   
#     for stock in mean_data.index:
#         if mean_data.loc[stock] < k:
            
#             lists.append(stock)
          
#     return lists

# def get_dataframe_prices (n, date_given, nan_threshold = 5,  volume_threshold = 150):
#     data = prices
    
#     """
#     Returns DataFrames of investible universe.
    
    
#     Parameters
#     ----------
#     n : int
#         How many previous n date points we want to look at.
#     date_given : str  
#         How many days ahead of the datetime index the target matrix will be set.
#     nan_threshold : int
#        What percentage of tolerance nan in the prices dataframe. 
#     volumes_threshold : int
#         Threshold of volumes
        
        
#     Returns
#     -------
#     Dataframe
    
#     """
    
    
#     n_5 = ( n * nan_threshold ) / 100
    
#     #finding the n date points ago date
#     array = data[data["date"] == date_given].index.values
#     n_days_ago = array[0] - n 
#     end_date = n + n_days_ago
#     #slicing the data up intil that date
#     data_1 = data.loc[n_days_ago+1: end_date]
#     #drop volumes that has low volumes
#     data_1 = data_1.drop(low_volumes(volume_threshold, n_days_ago+1, end_date), axis = 1)
#     #replace infinite value to nan
#     data_1.replace([np.inf, -np.inf], np.nan, inplace=True)   
#     data_date = data_1.set_index("date")
    
#     # prices_date.columns #calling all columns
#     for stock in data_date.columns: 
#         if data_1[stock].isnull().sum() > n_5:    #if the stock has 95% more stock value in the data
#             data_1 = data_1.drop([stock], axis = 1) #drop the stock that has n% more nan value
            
#     data_1 = data_1.set_index("date")
#     data_1.index = pd.to_datetime(data_1.index)
 
#     return data_1