#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from datetime import timedelta
import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

#Download Zipfile and create pandas DataFrame
FFdata = pd.read_csv(r'F-F_Research_Data_Factors.CSV',header = 0, names = ['Date','MKT-RF','SMB','HML','RF'], skiprows=3)

#Drop last row of data - String
FFdata = FFdata[:1074]

#Convert YYYYMM into Date
FFdata['Date'] = pd.to_datetime(FFdata['Date'], format = "%Y%m")
FFdata.index = FFdata['Date']
FFdata.drop(FFdata.columns[0], axis=1,inplace=True)

#Drop Days in YYYY-MM-DD
FFdata.index = FFdata.index.map(lambda x: x.strftime('%Y-%m'))

#Convert into float
FFdata = FFdata.astype('float')
print(FFdata.tail())

#Import Fund Data
#Tweedy, Browne Global Value Fund - Ticker TBGVX from YahooFinance

#Get Data from Yahoo
start = datetime.datetime(1980,1,1)
end = datetime.datetime(2016,1,1)
f = web.get_data_yahoo("TBGVX", start, end, interval='m')
#print(f.tail())
#Delete Columns
f.drop(f.columns[[0,1,2,3,5]], axis=1, inplace=True)
#print(f.tail())
#Fix Date Column
f.index = f.index.map(lambda x: x.strftime('%Y-%m'))
#print(f.head())
#Ln Return
f['LnReturn'] = np.log(f['Adj Close']) - np.log(f['Adj Close'].shift(1))
print(f.head())
#Merge DF
data2 = pd.concat([f,FFdata], axis = 1)

#Excess Return
data2['XRtrn'] = (data2['LnReturn']*100 - data2['RF'])
#Align Data
df = data2[np.isfinite(data2['XRtrn'])]

y = df['XRtrn']
X = df.ix[:,[2,3,4]]   #ix[] is an index function
X = sm.add_constant(X) #add constant in the model
model = sm.OLS(y, X)   #OLS regression
results = model.fit()
print(results.summary())

##plot
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(results, fig = fig)
plt.show()

#sns.lmplot(x="MKT-RF", y="XRtrn", data=data2)
