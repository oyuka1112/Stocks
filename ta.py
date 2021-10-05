#TODO Add functions that return a plot for each, or a boolean parameter that asks if we want to return a plot too

import pandas as pd
import numpy as np
import talib

"""
TA Functions

Parameters
----------
df : DataFrame
    The dataframe of the stock we want to analyze. (i.e. the y matrix from the get_market_Xy function)
timeperiod : int
    The last x amount of days
inplace: boolean
    If true, it will edit the dataframe passed in.
    If false, it will just return a copy of the dataframe + the new column

Returns
-------
The dataframe we passed in plus a new column containing the respective indicator we called (If inplace=True)
otherwise it returns nothing and our dataframe that we passed in is editted

"""

def add_ATR(df, timeperiod=14, inplace=False):
    if inplace:
        df['ATR_{}'.format(timeperiod)] = talib.ATR(df.iloc[:, 0], df.iloc[:, 0], df.iloc[:, 0], timeperiod=timeperiod)
    else:
        temp = df.copy()
        temp['ATR_{}'.format(timeperiod)] = talib.ATR(df.iloc[:, 0], df.iloc[:, 0], df.iloc[:, 0], timeperiod=timeperiod)
        return temp

def add_RSI(df, timeperiod=14, inplace=False):
    if inplace:
        df['RSI_{}'.format(timeperiod)] = talib.RSI(df.iloc[:, 0], timeperiod=timeperiod)
    else:
        temp = df.copy()
        temp['RSI_{}'.format(timeperiod)] = talib.RSI(df.iloc[:, 0], timeperiod=timeperiod)
        return temp

def add_SMA(df, timeperiod=200, inplace=False):
    if inplace:
        df['SMA_{}'.format(timeperiod)] = talib.SMA(df.iloc[:, 0], timeperiod)
    else:
        temp = df.copy()
        temp['SMA_{}'.format(timeperiod)] = talib.SMA(df.iloc[:, 0], timeperiod)
        return temp

def add_EMA(df, timeperiod=9, inplace=False):
    if inplace:
        df['EMA_{}'.format(timeperiod)] = talib.EMA(df.iloc[:, 0], timeperiod)
    else:
        temp = df.copy()
        temp['EMA_{}'.format(timeperiod)] = talib.EMA(df.iloc[:, 0], timeperiod)
        return temp

def add_OBV(df, volume, inplace=False):
    volume = volume*1000 # The volume in the df is in 1000's
    if inplace:
        df['OBV'] = talib.OBV(df.iloc[:, 0], volume)
    else:
        temp = df.copy()
        temp['OBV'] = talib.OBV(df.iloc[:, 0], volume)
        return temp