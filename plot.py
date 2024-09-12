import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
# file_path = 'your_file_path.csv'  # Replace with your file path

def plot(data):
    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Set up the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='Date', y='Prediction', hue='Ticker', marker='o', legend=False)

    # Place the stock ticker labels near their corresponding lines
    for ticker in data['Ticker'].unique():
        ticker_data = data[data['Ticker'] == ticker]
        plt.text(ticker_data['Date'].iloc[-1], ticker_data['Prediction'].iloc[-1], ticker, 
                horizontalalignment='left', size='medium', color='black', weight='semibold')

    # Customize the plot
    plt.title('Stock Predictions Over Last Week')
    plt.xlabel('Date')
    plt.ylabel('Prediction')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the plot
    plt.show()