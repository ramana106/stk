import pandas as pd
from utils import data_get_preprocess
from tickers import tickers

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
    df.to_csv("data.csv")
