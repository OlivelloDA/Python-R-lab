# -*- coding: utf-8 -*-
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

yf.pdr_override() # <== that's all it takes :-)
# download dataframe
adap_db = pdr.get_data_yahoo("ADAP", start="2020-01-01", end="2021-08-23")['Adj Close']
gold_db = pdr.get_data_yahoo("GC=F", start="2020-01-01", end="2021-08-23")['Adj Close']
commodity = pdr.get_data_yahoo("DJCI", start="2020-01-01", end="2021-08-23")['Adj Close']

data = pd.DataFrame({'adap' : adap_db, 'gold' : gold_db , 'commodity' : commodity}).reset_index(drop = True).dropna()
data_yesindex = pd.DataFrame({'adap' : adap_db, 'gold' : gold_db , 'commodity' : commodity}).dropna()

#data.corr(method = 'pearson')
#log-returns
data = data[['adap','gold','commodity']].pct_change().apply(lambda x: np.log(1+x))
data = data.drop([0])
data_yesindex= data_yesindex[['adap','gold','commodity']].pct_change().apply(lambda x: np.log(1+x))
data_yesindex = data_yesindex.dropna()


adap_vol = np.sqrt(data['adap'].var() * 250)
gold_vol = np.sqrt(data['gold'].var() * 250)
commodity_vol = np.sqrt(data['commodity'].var() * 250)
cov_matrix = data.cov()
corr_matrix = data.corr()

'''Note that we use the resample() function to get yearly returns. 
The argument to function, ‘Y’, denotes yearly.'''


ind_er = data_yesindex.mean()
w = [0.3 , 0.3 , 0.4]
port_er = (w*ind_er).sum()

ann_sd = data.std().apply(lambda x: x*np.sqrt(250))
assets = pd.concat([ind_er, ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
assets.columns = ['Log-Returns', 'Volatility']

p_ret = [] # Define an empty array for portfolio returns
p_vol = [] # Define an empty array for portfolio volatility
p_weights = [] # Define an empty array for asset weights

num_assets = len(data.columns)
num_portfolios = 10000
for portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights = weights/np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its 
                                      # weights 
    p_ret.append(returns)
    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
    sd = np.sqrt(var) # Daily standard deviation
    ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
    p_vol.append(ann_sd)


dataframe = {'Log-Returns':p_ret, 'Volatility':p_vol}

for counter, symbol in enumerate(data.columns.tolist()):
    #print(counter, symbol)
    dataframe[symbol+' weight'] = [w[counter] for w in p_weights]
portfolios  = pd.DataFrame(dataframe)
portfolios.head() # Dataframe of the 10000 portfolios created


portfolios.plot.scatter(x='Volatility', y='Log-Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10,10])
min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
# idxmin() gives us the minimum value in the column specified.                               


# plotting the minimum volatility portfolio
plt.subplots(figsize=[10,10])
plt.scatter(portfolios['Volatility'], portfolios['Log-Returns'],marker='o', s=10, alpha=0.3)
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='.', s=500)

# Finding the optimal portfolio
rf = 0.7 # risk factor
#highest sharp ratio
optimal_risky_port = portfolios.iloc[((portfolios['Log-Returns']-rf)/portfolios['Volatility']).idxmax()]
optimal_risky_port

# Plotting optimal portfolio
plt.subplots(figsize=(10, 10))
plt.scatter(portfolios['Volatility'], portfolios['Log-Returns'],marker='o', s=10, alpha=0.3)
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)










