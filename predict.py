import json
import numpy as np
import pandas as pd
import joblib
import traceback
from utils import data_get_preprocess, load_model, model_predict

def predict_tickers(tickers):

    # Initialize an empty DataFrame to accumulate results
    all_results = pd.DataFrame()

    features = json.load(open("features.json"))

    for ticker in tickers:
        try:
            data = data_get_preprocess(ticker, train=False)
            results = model_predict(data, features, ticker)
            all_results = pd.concat([all_results, results], ignore_index=True)
        except:
            traceback.print_exc()
            print(f"Error processing {ticker}. Skipping...")
            continue

    print("All results combined:")
    print(all_results)


if __name__ == "__main__":

    tickers = ['ARE&M.NS', 'ANGELONE.NS', 'CDSL.NS', 'DRREDDY.NS', 'HUDCO.NS', 'ITC.NS',
    'KESORAMIND.NS', 'KSCL.NS', 'MANAPPURAM.NS', 'NATCOPHARM.NS', 'SBICARD.NS',
    'SPANDANA.NS', 'TATACHEM.NS', 'TATAELXSI.NS', 'TATAMTRDVR.NS', 'TATASTEEL.NS', 'WABAG.NS']
    predict_tickers(tickers)