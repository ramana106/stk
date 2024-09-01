import yfinance as yf
import numpy as np
import pandas as pd
import pandas_ta as ta
import warnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from datetime import timedelta
import os
import sys
import tensorflow as tf
import contextlib
from yahooquery import Ticker

def get_fundamentals(ticker_symbol):
    # Use yahooquery to fetch fundamental data
    ticker = Ticker(ticker_symbol)
    summary_detail = ticker.summary_detail[ticker_symbol]
    financial_data = ticker.financial_data[ticker_symbol]
    balance_sheet = ticker.balance_sheet(frequency='quarterly').to_dict(orient='records')[-1]
    income_statement = ticker.income_statement(frequency='quaterly').to_dict(orient='records')[-1]
    key_stats = ticker.key_stats[ticker_symbol]

    market_cap = summary_detail.get('marketCap', np.nan)
    pe_ratio = summary_detail.get('trailingPE', np.nan)
    book_value = balance_sheet['TangibleBookValue'] / balance_sheet['OrdinarySharesNumber']

    operating_margin = financial_data.get('operatingMargins', np.nan)
    total_revenue = financial_data.get('totalRevenue', np.nan)
    invested_capital = balance_sheet.get('InvestedCapital', np.nan)
    ebit = operating_margin * total_revenue
    roce = ebit / invested_capital

    # Extract the necessary values
    total_debt = balance_sheet.get('TotalDebt')  # Total Debt
    equity = balance_sheet.get('StockholdersEquity')  # Stockholders' Equity

    # Calculate the Debt to Equity ratio
    debt_to_equity = total_debt / equity if equity else np.nan

    net_income = income_statement.get('NetIncome', 0)  # Net Income from the income statement
    equity = balance_sheet.get('StockholdersEquity', 1)  # Stockholders' Equity from the balance sheet

    # Calculate the ROE
    roe = (net_income / equity) * 100 if equity else None

    return market_cap, pe_ratio, book_value, roce, roe, debt_to_equity

# # Suppress TensorFlow and Keras warnings and logs
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.get_logger().setLevel('ERROR')
# tf.autograph.set_verbosity(0)
# warnings.filterwarnings('ignore')


# Function to suppress stdout and stderr
# @contextlib.contextmanager
# def suppress_stdout_stderr():
#     with open(os.devnull, 'w') as devnull:
#         old_stdout = sys.stdout
#         old_stderr = sys.stderr
#         sys.stdout = devnull
#         sys.stderr = devnull
#         try:
#             yield
#         finally:
#             sys.stdout = old_stdout
#             sys.stderr = old_stderr

# # Suppress TensorFlow progress bars
# @contextlib.contextmanager
# def suppress_tf_logs():
#     original_level = tf.get_logger().level
#     tf.get_logger().setLevel('ERROR')
#     yield
#     tf.get_logger().setLevel(original_level)

def save_model(model, filename='model.h5'):
    model.save(filename)

def load_model(filename='model.h5'):
    return tf.keras.models.load_model(filename)

def calculate_wave_trend(data, channel_length=9, avg_length=12):
    # Calculate the Typical Price
    data['Typical_Price'] = (data['High'] + data['Low'] + data['Adj Close']) / 3

    # Calculate the Exponential Moving Average (EMA) of the Typical Price
    data['esa'] = data['Typical_Price'].ewm(span=channel_length, adjust=False).mean()

    # Calculate the Exponential Moving Average of the Absolute Price Oscillator
    data['d'] = abs(data['Typical_Price'] - data['esa'])
    data['de'] = data['d'].ewm(span=channel_length, adjust=False).mean()

    # Calculate the WT Oscillator
    data['WT1'] = (data['Typical_Price'] - data['esa']) / (0.015 * data['de'])
    data['WT2'] = data['WT1'].ewm(span=avg_length, adjust=False).mean()

    return data

def ichimoku(data):
    # Calculate Ichimoku components
    nine_period_high = data['High'].rolling(window=9).max()
    nine_period_low = data['Low'].rolling(window=9).min()
    data['Ichimoku_Tenkan'] = (nine_period_high + nine_period_low) / 2

    period26_high = data['High'].rolling(window=26).max()
    period26_low = data['Low'].rolling(window=26).min()
    data['Ichimoku_Kijun'] = (period26_high + period26_low) / 2

    data['Ichimoku_SenkouA'] = ((data['Ichimoku_Tenkan'] + data['Ichimoku_Kijun']) / 2).shift(26)
    data['Ichimoku_SenkouB'] = ((data['High'].rolling(window=52).max() + data['Low'].rolling(window=52).min()) / 2).shift(26)
    data['Ichimoku_Chikou'] = data['Adj Close'].shift(-26)

    return data

def calculate_parabolic_sar(high, low, initial_acceleration=0.02, max_acceleration=0.2):
    """Calculate Parabolic SAR."""
    sar = np.zeros(len(high))
    trend = 1  # 1 for uptrend, -1 for downtrend
    ep = high.iloc[0]  # Extreme price
    af = initial_acceleration
    sar[0] = low.iloc[0]  # Initialize SAR

    for i in range(1, len(high)):
        if trend == 1:
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(max_acceleration, af + initial_acceleration)
            if low.iloc[i] < sar[i]:
                trend = -1
                sar[i] = ep
                ep = low.iloc[i]
                af = initial_acceleration
        else:
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(max_acceleration, af + initial_acceleration)
            if high.iloc[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = high.iloc[i]
                af = initial_acceleration

    return pd.Series(sar, index=high.index)


def download_stock_data(ticker_symbol):
    data = yf.download(ticker_symbol, period="max", interval="1d", progress=False)
    return data


def data_get_preprocess(ticker_symbol, st_idx=None, en_idx=None):
    print(f"\nProcessing {ticker_symbol}...")

    # Fetch historical data with suppressed progress bar
    data = download_stock_data(ticker_symbol)


    # Generate derived features
    data['Return'] = data['Adj Close'].pct_change() * 100
    data['Volatility'] = data['Return'].rolling(window=5).std()
    data['SMA_5'] = data['Adj Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Adj Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Adj Close'].rolling(window=20).mean()
    data['Momentum'] = data['Adj Close'] - data['Adj Close'].shift(4)
    data['RSI'] = 100 - (100 / (1 + data['Return'].rolling(window=14).mean() / data['Return'].rolling(window=14).std()))

    # Last week's features
    data['Last_Week_Return'] = data['Return'].rolling(window=5).sum()
    data['Last_Week_Volatility'] = data['Volatility'].rolling(window=5).mean()
    data['Last_Week_Momentum'] = data['Momentum'].rolling(window=5).mean()
    data['Last_Week_SMA'] = data['SMA_5'].rolling(window=5).mean()
    data['Last_Week_RSI'] = data['RSI'].rolling(window=5).mean()

    # Apply the wave trend function
    data = calculate_wave_trend(data)

    # WaveTrend Cross Lower Band (WT_CROSS_LB)
    data['WT_CROSS_LB'] = ((data['WT1'] < -60) & (data['WT2'] < -60)).astype(int)

    # Bollinger Bands
    data['BB_upper'] = data['SMA_20'] + 2 * data['Volatility']
    data['BB_lower'] = data['SMA_20'] - 2 * data['Volatility']

    # ATR
    data['ATR'] = data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min()

    # Williams %R
    data['Williams_%R'] = (data['High'].rolling(window=14).max() - data['Adj Close']) / (data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min()) * -100

    # Volume-Weighted Average Price
    data['VWAP'] = (data['Adj Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()

    # MACD
    data['EMA_5'] = data['Adj Close'].ewm(span=5, adjust=False).mean()
    data['EMA_10'] = data['Adj Close'].ewm(span=10, adjust=False).mean()
    data['EMA_12'] = data['Adj Close'].ewm(span=12, adjust=False).mean()
    data['EMA_20'] = data['Adj Close'].ewm(span=20, adjust=False).mean()
    data['EMA_26'] = data['Adj Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # OBV
    data['OBV'] = (np.sign(data['Adj Close'].diff()) * data['Volume']).cumsum()

    # Stochastic Oscillator
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    data['Stochastic_K'] = 100 * (data['Adj Close'] - low_14) / (high_14 - low_14)
    data['Stochastic_D'] = data['Stochastic_K'].rolling(window=3).mean()

    # Lag features
    data['Lag_1'] = data['Return'].shift(1)
    data['Lag_2'] = data['Return'].shift(2)
    data['Lag_3'] = data['Return'].shift(3)

    data['Parabolic_SAR'] = calculate_parabolic_sar(data['High'], data['Low'], initial_acceleration=0.02, max_acceleration=0.2)
    data['CCI'] = ta.cci(data['High'], data['Low'], data['Adj Close'], timeperiod=20)
    data['AD_Line'] = ta.ad(data['High'], data['Low'], data['Close'], data['Volume'])
    data['CMO'] = ta.cmo(data['Adj Close'], timeperiod=14)
    data = ichimoku(data)

    # Add fundamental data
    market_cap, pe_ratio, book_value, roce, roe, debt_to_equity = get_fundamentals(ticker_symbol)
    data['Market_Cap'] = market_cap
    data['P_E_Ratio'] = pe_ratio
    data['book_value'] = book_value
    data["roce"] = roce
    data["roe"] = roe
    data["debt_to_equity"] = debt_to_equity

    # Prepare the target variable (percentage change from previous day)
    data['Next_Day_Return'] = data['Return'].shift(-1)
    data['Next_Week_Return'] = ((data['Adj Close'].shift(-5) - data['Adj Close']) / data['Adj Close']) * 100
    
    # last_date = data.index.tolist()[-1]
    # print(f"last date : {last_date}")

    st_idx = 0 if st_idx is None else st_idx
    en_idx = len(data) if en_idx is None else en_idx

    data = data[st_idx:en_idx]
    
    # fill na with contant value
    constant_value = -9999
    data = data.fillna(constant_value)
    return data


def gen_process_data(features, data, pred_col="Next_Day_Return"):
    # Prepare data for modeling
    X = data[features]
    y = data[pred_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test, data

def create_model(input_dim=None, optimizer='adam', init='uniform', dropout_rate=0.3, neurons=256, loss='mean_squared_error'):
    if input_dim is None:
        raise ValueError("input_dim must be provided")

    # Build the model
    model = Sequential()
    # Use Input layer to define the input shape
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(neurons, kernel_initializer=init, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=['mean_absolute_error'])
    return model


def train(X_train, X_test, y_train, y_test):
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Instantiate the KerasRegressor
    model = KerasRegressor(
        model=create_model,
        input_dim=X_train.shape[1]
    )

    # Define the grid of hyperparameters to search
    param_grid = {
        'model__dropout_rate': [0.2, 0.3, 0.4],
        'model__neurons': [64, 128, 256],
        'model__init': ['uniform', 'normal'],
        'model__loss': ['mean_squared_error', 'mean_absolute_error', 'huber_loss'],
        'batch_size': [16, 32, 64],
        'epochs': [50, 100],
        'optimizer': ['adam', 'rmsprop']
    }

    # Perform Grid Search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
    grid_result = grid.fit(X_train, y_train)

    # Output the best score and parameters
    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

    # Save the best model found by GridSearchCV
    best_model = grid_result.best_estimator_.model
    save_model(best_model)

    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')


def model_predict(data, features, ticker):
    # Prepare data
    X_full = data[features]

    model = load_model()

    # Scale and predict using the full dataset
    scaler = joblib.load('scaler.pkl')
    X_full_scaled = scaler.transform(X_full)
    y_pred = model.predict(X_full_scaled).flatten()

    # Calculate a dynamic threshold based on the 99th percentile of predictions
    threshold = np.percentile(y_pred, 99)

    # Create a DataFrame to store results
    results = pd.DataFrame({
        'Date': data.index,
        'Ticker': ticker,
        'Prediction': y_pred,
        'Actual': data['Next_Week_Return'],  # Adjust this column name based on your data
        'Comparison': np.where((y_pred > threshold), 'Positive', 'Negative')
    })

    return results


def predict_tomorrow(data, features):
    scaler = joblib.load('scaler.pkl')
    model = load_model()

    # Scale the data
    last_row = data.iloc[-1][features].values.reshape(1, -1)
    last_row_scaled = scaler.transform(last_row)

    # Predict tomorrow's return
    with suppress_stdout_stderr():
        tomorrow_pred = model.predict(last_row_scaled).flatten()

    # Calculate a dynamic threshold based on the 99th percentile of the predictions
    with suppress_stdout_stderr():
        all_predictions_scaled = model.predict(scaler.transform(data[features]))
    threshold = np.percentile(all_predictions_scaled, 99)

    # Calculate tomorrow's date
    tomorrow_date = data.index[-1] + timedelta(days=1)

    # Print results
    great = False
    if tomorrow_pred[0] > threshold:
        great = True
    print(f"Predicted return on {tomorrow_date.date()}: {tomorrow_pred[0]:.4f} ---- >99th P {great}")



def test_strategy(X_train, X_test, y_train, y_test, y_pred, data):

    # Simulate trading strategy (same as before)
    initial_capital = 100000  # Starting with ₹100,000
    capital = 100000
    total_trades = 0  # To count the number of trades
    successful_trades = 0  # To count the number of successful trades

    # Calculate a dynamic threshold based on the 95th percentile of predictions
    threshold = np.percentile(y_pred, 99)

    for i in range(len(y_test)):
        if y_pred[i] > threshold:  # Invest only if prediction is greater than 2%
            # Buy at the start of the day
            buy_price = data['Adj Close'].iloc[len(X_train) + i]
            # Sell at the end of the day
            sell_price = data['Adj Close'].iloc[len(X_train) + i + 1]
            # Calculate the profit/loss from this trade
            profit_loss = capital * (sell_price - buy_price) / buy_price
            capital += profit_loss
            total_trades += 1  # Increment the trade count

            if profit_loss > 0:
                successful_trades += 1  # Increment the count of successful trades
            # else:
                # print(f"LOST > {profit_loss}")

    # Compare with buy-and-hold strategy
    buy_and_hold = initial_capital * (data['Adj Close'].iloc[len(X_train) + len(y_test) - 1] / data['Adj Close'].iloc[len(X_train)])

    # Calculate the duration of the test period in months
    test_start_date = data.index[len(X_train)]
    test_end_date = data.index[len(X_train) + len(y_test) - 1]
    duration_months = (test_end_date.year - test_start_date.year) * 12 + (test_end_date.month - test_start_date.month)

    # Output results
    print(f"Test period duration: {duration_months} months")
    print(f"Number of trades executed: {total_trades}")
    print(f"Number of successful trades: {successful_trades}")
    print(f"Final capital with trading strategy: ₹{capital:.2f}")
    print(f"Final capital with buy-and-hold strategy: ₹{buy_and_hold:.2f}")
    print(f"Profit or Loss: ₹{capital - initial_capital:.2f}")
