import pandas as pd
from utils import data_get_preprocess
from tickers import tickers
import traceback
import numpy as np

def save_data():
    df = pd.DataFrame()
    for ticker in tickers:
        try:
            data = data_get_preprocess(ticker, en_idx=-30)
            df = pd.concat([df, data])
        except:
            traceback.print_exc()
            print(f"Error processing {ticker}. Skipping...")
            continue
    print(len(df))
    df.replace([np.inf, -np.inf], -9999, inplace=True)
    df.to_csv("data.csv")
