import multiprocessing as mp
import pandas as pd
from tqdm import tqdm
from diff_cap_packages import stock_selection, models_
from pycaret.regression import *
from itertools import product
from time import time


def get_predictions(lrmodels, from_date, days=500):
    investible_universe = models_.get_investible_universe(from_date, days)

    common_features = models_.get_common_features(from_date, days)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = tqdm(pool.map(prediction_helper, product([common_features], investible_universe, [lrmodels], [from_date], [days])))

    predictions = pd.concat(results, axis=0)
    return predictions


def prediction_helper(p):
    common_features, stock, linmodels, from_date, days = p[0], p[1], p[2], p[3], p[4]

    predictions = pd.DataFrame(index=[stock], columns=["predicted % chg"])
    data = models_.get_dataframe_for_pycaret(stock, from_date, days, common_features)
    model = linmodels.loc[stock, "linear regression model"]
    pred_pct_chg = stock_selection.pct_change(predict_model(model, data).loc[from_date, "Label"], data.loc[from_date, stock])
    predictions.loc[stock, "predicted % chg"] = pred_pct_chg
    act_pct_chg = stock_selection.pct_change(data.loc[from_date, stock + " +1 day"], data.loc[from_date, stock])
    predictions.loc[stock, "actual % chg"] = act_pct_chg

    return predictions


if __name__=="__main__":
    lrmodels = stock_selection.load_models("2021-06-29")

    a = time()
    predictions = get_predictions(lrmodels, "2021-06-29", 500)
    b = time()

    print(predictions)
    print("Elapsed time (sec.):", b - a)
