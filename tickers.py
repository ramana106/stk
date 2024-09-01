import yfinance as yf

def is_valid_ticker(ticker):
    stock = yf.Ticker(ticker)
    # Check if the stock info is available
    return stock.info['regularMarketPrice'] is not None

tickers = [
    # Financial Services
    "ANGELONE.NS", "HDFCBANK.NS", "IDFC.NS", "KOTAKBANK.NS", "KTKBANK.NS", 
    "TMB.NS", "UJJIVANSFB.NS", "SBICARD.NS", "ARMANFIN.NS", "AAVAS.NS",

    # Pharmaceuticals & Healthcare
    "DRREDDY.NS", "NATCOPHARM.NS", "SUNPHARMA.NS", "CIPLA.NS", "DIVISLAB.NS",
    
    # Industrial Manufacturing
    "HUDCO.NS", "KESORAMIND.NS", "TATAMTRDVR.NS", "TATASTEEL.NS", 
    "WABAG.NS", "MIDHANI.NS", "SOLARINDS.NS", "MAZDOCK.NS", 
    "COCHINSHIP.NS", "GRSE.NS", "HAL.NS", "ASTRAMICRO.NS",
    "DATAPATTNS.NS", "BEL.NS", "BDL.NS", "PARAS.NS", "INDUSINDBK.NS",

    # Chemicals
    "TATACHEM.NS", "UPL.NS", "BASF.NS",

    # Consumer Goods
    "ITC.NS", "CDSL.NS", "MANAPPURAM.NS", "HINDUNILVR.NS", "DABUR.NS",
    
    # Technology
    "TATAELXSI.NS", "ZENTEC.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS",

    # Energy
    "RELIANCE.NS", "ONGC.NS", "NTPC.NS", "ADANIGREEN.NS", "POWERGRID.NS",

    # Utilities
    "GAIL.NS", "INDIANB.NS", "TATAPOWER.NS", "ADANITRANS.NS",

    # Telecom
    "BHARTIARTL.NS", "RELIANCE.NS", "MTNL.NS",

    # Materials
    "JSWSTEEL.NS", "HINDALCO.NS", "NALCO.NS",

    # Other
    "ARE&M.NS", "SPANDANA.NS", "VBL.NS", "ZEEL.NS", "TV18BRDCST.NS"
]

if __name__ == "__main__":
    for ticker in tickers:
        if is_valid_ticker(ticker):
            print(f"{ticker} is valid.")
        else:
            print(f"{ticker} is invalid.")