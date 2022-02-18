### Multi-layer Perceptron  Version 1
### Multi Layer Perceptionn to predict next time step

import datetime as dt
import numpy as np
import pandas as pd 
from sklearn import neural_network
from sklearn.model_selection import GridSearchCV

def MLP(estacion,gcm,cal_date,app_date):
    c=time.time()
    #Pre-processing Data
    start=estacion.index[0]
    future=pd.DataFrame(index=gcm.loc[start:].index) 
    future[estacion.name]=estacion.loc[:cal_date] 
    mcg=np.reshape(gcm.loc[start:cal_date].values,(len(gcm.loc[start:cal_date].values),1)) 
    fut=np.array(future[estacion.name][:cal_date].values, dtype='float')      
    ##Creating Predictive Model
    # hls_list =[(int(a),) for a in np.linspace(10,50,5)]
    parameters = {
        "hidden_layer_sizes": [(200,)],
        "activation": ["identity","logistic"],
        "solver": ["adam"],
        "learning_rate":["adaptive"],
        "learning_rate_init":[0.0001]
        }
    model = GridSearchCV(neural_network.MLPRegressor(), parameters, cv=20, verbose=0,scoring='neg_median_absolute_error') 
    model.fit(mcg,fut) 
    #Predicting
    dates=gcm.loc[cal_date+dt.timedelta(days=1):app_date].index
    fut_gcm=np.reshape(gcm['GCM'].loc[dates].values,(dates.shape[0],1))
    future.loc[dates]=np.reshape(model.predict(fut_gcm),(dates.shape[0],1))
    data_anually=future.resample('A', how='sum')
    data_anually.plot()
    print(model.best_params_)
    print(model.best_score_)
    tim=(time.time()-c)/60
    return future

    model.best_params_
    model.best_score_