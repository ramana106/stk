import pandas as pd
import json
from utils import data_get_preprocess, gen_process_data, train, train_grid

def func(tickers, st_idx, en_idx, tuning_pram=None):
    features = [
                'Return', 'Volatility', 'SMA_5', 'SMA_10', 'SMA_20', 'Momentum', 'RSI',
                'BB_upper', 'BB_lower', 'MACD', 'Signal_Line', 'Stochastic_K', 'Stochastic_D',
                'Lag_1', 'Lag_2', 'Lag_3', 'Volume', 'WT_CROSS_LB',
                'WT1', 'WT2', 'ATR', 'Williams_%R',
                'VWAP', 'OBV', 'Parabolic_SAR', 'CCI', 'AD_Line', 'CMO',
                'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_SenkouA', 'Ichimoku_SenkouB', 'Ichimoku_Chikou',
                'Last_Week_Return', 'Last_Week_Volatility', 'Last_Week_Momentum', 'Last_Week_SMA', 'Last_Week_RSI',
                "Market_Cap", "P_E_Ratio", "book_value", "roce", "roe", "debt_to_equity"
            ]
    with open("features.json", 'w') as d:
        d.write(json.dumps(features))

    # tickers = [tickers[0]]
    # Process each ticker symbol
    import traceback
    df = pd.DataFrame()
    for ticker in tickers:
        try:
            data = data_get_preprocess(ticker, st_idx=st_idx, en_idx=en_idx)
            df = pd.concat([df, data])
        except:
            traceback.print_exc()
            print(f"Error processing {ticker}. Skipping...")
            continue
    print(len(df))
    X_train, X_test, y_train, y_test, data = gen_process_data(features, df, pred_col="Next_Week_Return")

    if tuning_pram:
        train_grid(X_train, X_test, y_train, y_test, tuning_pram)
    else:
        train(X_train, X_test, y_train, y_test)


if __name__ == "__main__":

    from tickers import tickers
    param_grid = {
        'model__dropout_rate': [0.2, 0.3, 0.4],
        'model__neurons': [64, 128, 256],
        'model__init': ['uniform', 'normal'],
        'model__loss': ['mean_squared_error', 'mean_absolute_error', 'huber_loss'],
        'batch_size': [16, 32, 64],
        'epochs': [50, 100],
        'optimizer': ['adam', 'rmsprop']
    }
    func(tickers, 0, -30, param_grid)