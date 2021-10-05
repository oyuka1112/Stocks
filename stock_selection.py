import os
from sys import exit
import shutil
from diff_cap_packages import Xy, models_
from tqdm import tqdm
from pycaret.regression import *
import pandas as pd
from itertools import product
import multiprocessing as mp
from sklearn.linear_model import LinearRegression
import pickle

def make_and_store_linear_models(end_date, num_days, force_overwrite=False, pycaret=True):
    suffix = "" if pycaret else "-npc"
    parent_path = os.path.join('models', end_date + "-linear-" + str(num_days) + suffix)
    investible_universe = models_.get_investible_universe(end_date, num_days)
    
    if os.path.isfile(os.path.join(parent_path, 'completed.txt')):
        print("This set of models is completely trained already.")
        print("Set force_overwrite to True if you want to overwrite it.")
    else:
        os.makedirs(parent_path, exist_ok=True)
        shutil.rmtree(parent_path)
        
    os.makedirs(parent_path, exist_ok=force_overwrite)
    
    try:
        common_features = models_.get_common_features(end_date, num_days)
        with mp.Pool(processes=mp.cpu_count()) as pool:
            list(tqdm(pool.imap(make_and_store_helper, product([common_features], investible_universe, [end_date], [num_days], [parent_path], [pycaret])), total=len(investible_universe)))
        
        with open(os.path.join(parent_path, 'completed.txt'), 'w') as file:
            file.write("This set of models is completely trained.")
            
    except Exception as e:
        print("Error:", e)
        print("Deleting new directory at:", parent_path)
        print("An error occured.")
        shutil.rmtree(parent_path)
        exit()
        
def make_and_store_helper(p):
    common_features, stock, end_date, num_days, parent_path, pycaret = p
    data = models_.get_dataframe_for_pycaret(stock, end_date, num_days, common_features)
    path_to_save = os.path.join(parent_path, stock)
    if pycaret:
        models_.setup_pycaret(data)
        model = create_model("lr")
        save_model(model, path_to_save, verbose=False)
    else:
        model = LinearRegression()
        model.fit(data.iloc[1:-1, :-1].fillna(method="bfill"), data.iloc[1:-1, -1])
        pickle.dump(model, open(path_to_save + ".pkl", 'wb'))
    
    
def load_models(end_date, num_days, pycaret=True):
    suffix = "" if pycaret else "-npc"
    parent_path = os.path.join('models', end_date + "-linear-" + str(num_days) + suffix)
    investible_universe = models_.get_investible_universe(end_date, num_days)
    lrmodels = pd.DataFrame(index=investible_universe)

    for stock in tqdm(investible_universe):
        model_path = os.path.join(parent_path, stock)
        if pycaret:
            lrmodels.loc[stock, "linear regression model"] = load_model(model_path, verbose=False)
        else:
            lrmodels.loc[stock, "linear regression model"] = pickle.load(open(model_path + ".pkl", 'rb'))
        
    return lrmodels

def get_predictions(lrmodels, end_date, num_days, message=False, pycaret=True):
    investible_universe = models_.get_investible_universe(end_date, num_days)
    suffix = "" if pycaret else "-npc"
    parent_path = os.path.join('models', end_date + "-linear-" + str(num_days) + suffix)
    
    if os.path.isfile(os.path.join(parent_path, 'predictions.pkl')):
        if message:
            print("Loading saved predictions...")
        return pd.read_pickle(os.path.join(parent_path, 'predictions.pkl'))
    else:
        print("Predictions not detected. Calculating and saving...")
        common_features = models_.get_common_features(end_date, num_days)
        common_features = common_features.loc[[common_features.index[-1]]]
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap(prediction_helper, product([common_features], investible_universe, [lrmodels], [end_date], [num_days], [pycaret])), total=len(investible_universe)))
            
        predictions = pd.concat(results, axis=0)
        predictions.to_pickle(os.path.join(parent_path, 'predictions.pkl'))
        return predictions


def prediction_helper(p):
    common_features, stock, linmodels, end_date, num_days, pycaret = p
    predictions = pd.DataFrame(index=[stock], columns=["predicted % chg"])
    model = linmodels.loc[stock, "linear regression model"]
    if pycaret:
        data = models_.get_dataframe_for_pycaret(stock, end_date, num_days, common_features).iloc[[-1]]
        pred_pct_chg = pct_change(predict_model(model, data).loc[end_date, "Label"], data.loc[end_date, stock])
    else:
        data = models_.get_dataframe_for_pycaret(stock, end_date, num_days, common_features).iloc[[-1]].iloc[:, :-1]
        pred_pct_chg = pct_change(model.predict(data)[0], data.loc[end_date, stock])
    predictions.loc[stock, "predicted % chg"] = pred_pct_chg
    act_pct_chg = pct_change(Xy.get_y(stock, target_ahead_by=1).loc[end_date, stock + ' +1 day'], data.loc[end_date, stock])
    predictions.loc[stock, "actual % chg"] = act_pct_chg

    return predictions
        
def pct_change(future, now):
    return (future - now)/now

def get_buy_sell(predictions, abs_vol_thresh = 0.08, how_many=5):
    predictions_filtered = predictions[predictions["predicted % chg"].abs() < abs_vol_thresh]
    predictions_sorted = predictions_filtered.sort_values("predicted % chg", ascending=False)
    
    buy = predictions_sorted.head(how_many)
    buy["predicted action"] = "BUY"
    sell = predictions_sorted.tail(how_many)
    sell["predicted action"] = "SELL"
    buy_sell = pd.concat([buy, sell])
    
    return buy_sell
    
def get_buys_from_buy_sell(buy_sell):
    return buy_sell.head(int(len(buy_sell)/2)).index.values

def get_sells_from_buy_sell(buy_sell):
    return buy_sell.tail(int(len(buy_sell)/2)).index.values

def get_buy_sell_analysis(buy_sell, lenient_on_holds=True):
    analysis = buy_sell.copy()
    
    for stock in tqdm(analysis.index):
        if analysis.loc[stock, "actual % chg"] > 0:
            analysis.loc[stock, "proper action"] = "BUY"
        elif analysis.loc[stock, "actual % chg"] < 0:
            analysis.loc[stock, "proper action"] = "SELL"
        else:
            analysis.loc[stock, "proper action"] = "HOLD"

        if lenient_on_holds:
            analysis.loc[stock, "ACCURATE (HOLD = True)"] = (analysis.loc[stock, "proper action"] == analysis.loc[stock, "predicted action"]) or (analysis.loc[stock, "proper action"] == "HOLD")
        else:
            analysis.loc[stock, "ACCURATE (HOLD = False)"] = (analysis.loc[stock, "proper action"] == analysis.loc[stock, "predicted action"])
    
    return analysis
                        
def get_actual_overall_pct_chg(buy_sell):
    half_len = int(len(buy_sell)/2)
    return (buy_sell.head(half_len)["actual % chg"].mean() - buy_sell.tail(half_len)["actual % chg"].mean()) / 2
                        
def get_predicted_overall_pct_chg(buy_sell):
    half_len = int(len(buy_sell)/2)
    return (buy_sell.head(half_len)["predicted % chg"].mean() - buy_sell.tail(half_len)["predicted % chg"].mean()) / 2
    
