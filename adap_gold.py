import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)
# download dataframe
adap_db = pdr.get_data_yahoo("ADAP", start="2021-01-01", end="2021-08-22")['Adj Close']
gold_db = pdr.get_data_yahoo("GC=F", start="2021-01-01", end="2021-08-22")['Adj Close']

data = pd.DataFrame({'adap' : adap_db, 'gold' : gold_db})

data.corr(method = 'pearson')



















gold = yf.Ticker("GC=F")#gold future december 21
# get stock info
usdeur = yf.Ticker("USDEUR=X")
#usd/eur 

apple = yf.Ticker("AAPL")#Apple
apple
apple.info

# get historical market data
hist = apple.history(period="max")

# show actions (dividends, splits)
apple.actions

# show dividends
apple.dividends

# show splits
apple.splits

# show financials
apple.financials
apple.quarterly_financials

# show major holders
apple.major_holders

# show institutional holders
apple.institutional_holders

# show balance sheet
apple.balance_sheet
apple.quarterly_balance_sheet

# show cashflow
apple.cashflow
apple.quarterly_cashflow

# show earnings
apple.earnings
apple.quarterly_earnings

# show sustainability
apple.sustainability

# show analysts recommendations
apple.recommendations

# show next event (earnings, etc)
apple.calendar

# show ISIN code - *experimental*
# ISIN = International Securities Identification Number
apple.isin

# show options expirations
apple.options

# get option chain for specific expiration
opt = apple.option_chain('YYYY-MM-DD')
# data available via: opt.calls, opt.puts