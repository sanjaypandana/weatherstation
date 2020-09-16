#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 01:11:28 2019

@author: SanjayPandana
"""
import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

dataset = pd.read_csv('data.csv')
dataset.head()

columns = ["Station_Code","isSWMP","Historical","ProvisionalPlus","Frequency","F_Record","F_ATemp","F_RH","BP","F_BP","WSpd","F_WSpd","MaxWSpd","F_MaxWSpd","MaxWSpdT","Wdir","F_Wdir","SDWDir","F_SDWDir","TotPAR","F_TotPAR","TotPrcp","F_TotPrcp"]

dataset.drop(columns, inplace=True, axis=1)

dataset.drop(dataset.columns[[-1,]], axis=1, inplace=True)
dataset.rename(columns={'DateTimeStamp': 'DateTime', 'ATemp': 'Temperature','RH': 'Humidity'}, inplace=True)

df = dataset
df.head()

df['DateTime'] = pd.to_timedelta(pd.to_datetime(df['DateTime']), unit='ns').dt.total_seconds().astype(int)
#df['DateTime'] = pd.to_datetime(dataset['DateTime'])
df.head()

df['DateTime'] = pd.to_datetime(df['DateTime'],unit='s')
df.head()
df = df.set_index('DateTime')
df.head()

temp = df
temp = temp.iloc[:, 0:1]
temp.head()

hum = df
hum = hum.iloc[:, 1:2]
hum.head()

plt.rcParams.update({'font.size': 65})
temp.plot(figsize=(100,50),color='red',markersize=1,title='Temperature')
plt.show()

plt.rcParams.update({'font.size': 65})
hum.plot(figsize=(100,50),markersize=1,title='Humidity',color='blue')
plt.show()

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


############    Temp

warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod_temp = sm.tsa.statespace.SARIMAX(temp,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results_temp = mod_temp.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results_temp.aic))
        except:
            continue
        
        
mod_temp = sm.tsa.statespace.SARIMAX(temp,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results_temp = mod_temp.fit()

print(results_temp.summary().tables[1])
        
        
############    Hum

warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod_hum = sm.tsa.statespace.SARIMAX(hum,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results_hum = mod_hum.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results_hum.aic))
        except:
            continue
        
        
        
mod_hum = sm.tsa.statespace.SARIMAX(hum,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results_hum = mod_hum.fit()

print(results_hum.summary().tables[1])