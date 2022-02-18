### Random Forest Tree Version 2
### Use tree decision clasifiers 
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def Decision_tree(estacion,gcm,cal_date,app_date):
    #Pre-processing Data
    start=estacion.index[0]
    future=pd.DataFrame(index=gcm.loc[start:].index) 
    future[estacion.name]=estacion.loc[:cal_date]
    mcg=np.reshape(gcm.loc[start:cal_date].values,(len(gcm.loc[start:cal_date].values),1)) 
    fut=np.array(future[estacion.name][:cal_date].values, dtype='float')      
    ##Creating Predictive Model
    parameters = {
        "n_estimators": [10,50,100],
        "min_samples_split": [2],
        "min_samples_leaf": [1] 
        }
    model = GridSearchCV(RandomForestRegressor(), parameters, cv=25, verbose=0,scoring='neg_median_absolute_error') 
    model.fit(mcg,fut) 
    #Predicting
    dates=gcm.loc[cal_date+dt.timedelta(days=1):app_date].index
    fut_gcm=np.reshape(gcm['GCM'].loc[dates].values,(dates.shape[0],1))
    future.loc[dates]=np.reshape(model.predict(fut_gcm),(dates.shape[0],1))
    return future
