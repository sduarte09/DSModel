### Linear regression Version 2
### Linear regression to predict next time step
import numpy as np # Array processing for numbers, strings, records, and objects
import ML_Functions.Chaos_Statistics
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def SVRegression(x,y,series,dynamics,steps):
    parameters = {
    "kernel": ['sigmoid','rbf'],
    "C": [1,10,100],
    "gamma": [1e-3,1e-5,1e-7],
    "epsilon": [np.percentile(abs(series[1:]-series[:-1]),80)]
    }
    clf = GridSearchCV(SVR(), parameters, cv=3, verbose=0)    
    clf.fit(x,y) 
    for i in range(steps):  
      act=ML_Functions.Chaos_Statistics.construct(series, dynamics[1], dynamics[2]) 
      series=np.append(series,clf.predict(act))
    return series[-steps:]

#def SVRegression_1(series,steps):
#    lag=ML_Functions.Chaos_Statistics.delay(series)
#    x1=series[:-lag]
#    x2=series[lag-1:-1]
#    x=np.transpose(np.vstack((x1, x2)))
#    ynum = series[lag:]
#    parameters = {
#    "kernel": ["rbf"],
#    "C": [0.1,1,10,100],
#    "gamma": [1e-1,1e-3,1e-5,1e-7],
#    "epsilon": [np.percentile(abs(series[1:]-series[:-1]),75)]
#    }
#    clf = GridSearchCV(SVR(), parameters, cv=5, verbose=0)    
#    clf.fit(x,ynum) 
#    for i in range(steps):  
#      act=np.array([series[-lag],series[-1]]).reshape(1,2)   
#      series=np.append(series,clf.predict(act))
#    return series[-steps:]
#
#def SVRegression_2(series,steps):
#    lag=ML_Functions.Chaos_Statistics.delay(series)
#    x1=series[:-lag]
#    x2=series[lag-1:-1]
#    x=np.transpose(np.vstack((x1, x2)))
#    ynum = series[lag:]
#    parameters = {
#    "kernel": ["sigmoid"],
#    "C": [1,10,100],
#    "gamma": [1e-3,1e-5,1e-7],
#    "epsilon": [np.percentile(abs(series[1:]-series[:-1]),75)]
#    }
#    clf = GridSearchCV(SVR(), parameters, cv=5, verbose=0)    
#    clf.fit(x,ynum) 
#    for i in range(steps):  
#      act=np.array([series[-lag],series[-1]]).reshape(1,2)   
#      series=np.append(series,clf.predict(act))
#    return series[-steps:]
#
#def SVRegression_3(series,steps):
#    lag=ML_Functions.Chaos_Statistics.delay(series)
#    x1=series[:-lag]
#    x2=series[lag-1:-1]
#    x=np.transpose(np.vstack((x1, x2)))
#    ynum = series[lag:]
#    parameters = {
#    "kernel": ["poly"],
#    "C": [1,10,100],
#    "degree":[3,5,7,9],
#    "gamma": [1e-3,1e-5,1e-7],
#    "epsilon": [np.percentile(abs(series[1:]-series[:-1]),75)]
#    }
#    clf = GridSearchCV(SVR(), parameters, cv=5, verbose=0)    
#    clf.fit(x,ynum) 
#    for i in range(steps):  
#      act=np.array([series[-lag],series[-1]]).reshape(1,2)   
#      series=np.append(series,clf.predict(act))
#    return series[-steps:]


#import time
#sta=time.time()
#aaa=SVRegression(series,steps)
#from matplotlib import pyplot
#pyplot.plot(validate.values)
#pyplot.plot(aaa[:720], color='red')
#pyplot.show()
#fin=(time.time()-sta)/60


