### K- Nearest Neighbors Version 3
### Find the k Nearest negihbors to predict next time step
import datetime as dt
import numpy as np
import pandas as pd 
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV

def K_NN(estacion,gcm,cal_date,app_date):
    #Pre-processing Data
    start=estacion.index[0]
    future=pd.DataFrame(index=gcm.loc[start:].index) 
    future[estacion.name]=estacion.loc[:cal_date] 
    mcg=np.reshape(gcm.loc[start:cal_date].values,(len(gcm.loc[start:cal_date].values),1)) 
    fut=np.array(future[estacion.name][:cal_date].values, dtype='float')      
    ##Creating Predictive Model
    parameters = {
        "weights": ["uniform"],
        "algorithm": ["ball_tree"],
        "n_neighbors": [1,2,3],
        "leaf_size":[5,10,15,20,25,30]
        }
    model = GridSearchCV(neighbors.KNeighborsRegressor(), parameters, cv=30, verbose=0,scoring='neg_median_absolute_error') 
    model.fit(mcg,fut) 
    #Predicting
    dates=gcm.loc[cal_date+dt.timedelta(days=1):app_date].index
    fut_gcm=np.reshape(gcm['GCM'].loc[dates].values,(dates.shape[0],1))
    future.loc[dates]=np.reshape(model.predict(fut_gcm),(dates.shape[0],1))
    return future