import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from diff_cap_packages import Xy, filters, ta
from pycaret.regression import *

def get_investible_universe(end_date, num_days):
    end_date = pd.to_datetime(end_date)
    return filters.get_dataframe_prices(end_date, num_days, no_NaN_threshold=num_days).columns.values

def get_common_features(end_date, num_days):
    end_date = pd.to_datetime(end_date)
    return filters.get_dataframe_prices(end_date, num_days, no_NaN_threshold=num_days)
    
def get_accurate_predictions(analysis_matrix):
    total = analysis_matrix.shape[0]
    accurate = analysis_matrix['accurate'].sum()
    proportion_correct = accurate / total
    return proportion_correct, (str(accurate) + " correct out of " + str(total) + " || Proportion = " + str(proportion_correct))
    
def get_analysis(model, stock, features, target):
    predicted = predict_with_model(model, features).loc[:, ["Label"]]
    results = pd.concat([features.loc[:, stock], target, predicted], axis=1)
    results.columns = [stock, 'actual next day', 'predicted next day']
    results["actual % chg"] = (results["actual next day"] - results[stock]) / results[stock]
    results["pred % chg"] = (results["predicted next day"] - results[stock]) / results[stock]
    results["% error"] = (results["pred % chg"] - results["actual % chg"]) / results["actual % chg"]
    results["buy?"] = results["actual % chg"] > 0
    results["pred buy?"] = results["pred % chg"] > 0
    results['accurate'] = results["buy?"] == results["pred buy?"]
    return results
    
def predict_with_model(model, features):
    predicted = predict_model(model, features)
    return predicted
    
def setup_pycaret(data):
    setup(data=data, target=data.columns[-1], silent=True)
    
def get_best_model(data):
    return compare_models()
    
def get_dataframe_for_pycaret(stock_id, end_date, num_days, common_features=None):
    if common_features is None:
        print("Get common_features using models_.get_common_features")
    
    technical_features = get_technical_features_for_stock(stock_id, end_date, num_days)
    target = get_target_for_stock(stock_id, end_date, num_days)
    result = pd.concat([common_features, technical_features, target], axis=1)
    result.iloc[-1,-1] = np.nan
    return result
        
def get_technical_features_for_stock(stock_id, end_date, num_days):
    end_date = pd.to_datetime(end_date)
    technicals = Xy.get_y(stock_id, target_ahead_by=0)
    volume = filters.volumes.loc[:, stock_id]

    ta.add_ATR(technicals, inplace=True)
    ta.add_OBV(technicals, volume, inplace=True)
    # ta.add_RSI(technicals, inplace=True)

    technicals.drop(stock_id, axis=1, inplace=True)
    relevant_date_technicals = technicals.loc[:end_date].tail(num_days)

    return relevant_date_technicals
    
def get_target_for_stock(stock_id, end_date, num_days, target_ahead_by=1):
    end_date = pd.to_datetime(end_date)
    target = Xy.get_y(stock_id, target_ahead_by=target_ahead_by)
    target = target.loc[:end_date].tail(num_days)
    return target
