### BASIC STATISTICS
### Mean, Minimum, Maximum, etc...

import math # Mathematic operations
import numpy as np # Array processing for numbers, strings, records, and objects
import pandas as pd


def Statics(data):
### Daily Mean Data
  daily_mean=data.resample('D', how='mean')
  miss_data=daily_mean['close'].loc[daily_mean['close'].isnull()].index
  daily_mean=daily_mean.drop(miss_data)
#  daily_mean['range'] = daily_mean['max'] -daily_mean['min'] 
#  daily_mean['close-max'] = abs(daily_mean['close'] -daily_mean['max'])
#  daily_mean['close-min'] = abs(daily_mean['close'] -daily_mean['min']) 
#  daily_mean['close-open'] = abs(daily_mean['close'] -daily_mean['open'])
  daily_mean=daily_mean.add_suffix('_mean')
### Daily Min Data
  daily_min=data.resample('D', how='min')
  daily_min=daily_min.drop(miss_data)
#  daily_min['range'] = daily_min['max'] -daily_min['min'] 
#  daily_min['close-max'] = abs(daily_min['close'] -daily_min['max'])
#  daily_min['close-min'] = abs(daily_min['close'] -daily_min['min']) 
#  daily_min['close-open'] = abs(daily_min['close'] -daily_min['open'])
  daily_min=daily_min.add_suffix('_min')
### Daily Max Data
  daily_max=data.resample('D', how='max')
  daily_max=daily_max.drop(miss_data)
#  daily_max['range'] = daily_max['max'] -daily_max['min'] 
#  daily_max['close-max'] = abs(daily_max['close'] -daily_max['max'])
#  daily_max['close-min'] = abs(daily_max['close'] -daily_max['min']) 
#  daily_max['close-open'] = abs(daily_max['close'] -daily_max['open'])
  daily_max=daily_max.add_suffix('_max')
### Daily Mean Data
  daily_close=data.resample('D', how='last')
  daily_close=daily_close.drop(miss_data)
  daily_close=daily_close.add_suffix('_close')
### Join DataFrames 
  result = pd.concat([daily_mean,daily_min,daily_max,daily_close], axis=1, sort=False)
  result['close_range'] = abs(result['close_max'] -result['close_min'])
  result=result.reindex_axis(sorted(result.columns), axis=1)
  return result
  
  
  