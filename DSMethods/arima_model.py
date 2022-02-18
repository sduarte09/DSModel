### K- Nearest Neighbors Version 1
### Find the k Nearest negihbors to predict next time step
from matplotlib import pyplot
import statsmodels.api as sm
import itertools
import sys
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
import time
warnings.filterwarnings("ignore")

## evaluate an ARIMA model for a given order (p,d,q)
#def evaluate_arima_model(X, arima_order):
#	# prepare training dataset
#	train_size = int(len(X) * 0.66)
#	train, test = X[0:train_size], X[train_size:]
#	history = [x for x in train]
#	# make predictions
#	predictions = list()
#	for t in range(len(test)):
#		model = ARIMA(history, order=arima_order)
#		model_fit = model.fit(disp=0)
#		yhat = model_fit.forecast()[0]
#		predictions.append(yhat)
#		history.append(test[t])
#	# calculate out of sample error
#	error = mean_squared_error(test, predictions)
#	return error
#
#def evaluate_models(dataset, p_values, d_values, q_values):
#	dataset = dataset.astype('float32')
#	best_score, best_cfg = float("inf"), None
#	for p in p_values:
#		for d in d_values:
#			for q in q_values:
#				order = (p,d,q)
#				try:
#					mse = evaluate_arima_model(dataset, order)
#					if mse < best_score:
#						best_score, best_cfg = mse, order
#					print('ARIMA%s MSE=%.3f' % (order,mse))
#				except:
#					continue
#	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))   
#	return best_cfg
#
#def arima(series,steps):
#    prediction=[]
#    history = [x for x in series]
#    p_values = [0, 1, 2, 4, 6, 8, 10]
#    d_values = range(0, 3)
#    q_values = range(0, 3)
#    cfg=evaluate_models(series[-1000:], p_values, d_values, q_values)
#    for t in range(0,steps):
#      	model = ARIMA(history, order=cfg)
#      	model_fit = model.fit(disp=0)
#      	output = model_fit.forecast()
#      	prediction.append(output[0])
#    return prediction
#
#start=time.time()
#predict=arima(series,steps)
#duration=(time.time()-start)/60
#
#
#evaluate_arima_model(series[-1000:],(1,1,1))


warnings.filterwarnings("ignore")
# define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 3)
# generate all different combinations of p, d and q triplets
pdq = list(itertools.product(p, d, q))
# generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 60) for x in list(itertools.product(p, d, q))]
# choose the model for which the fitted data results in the lowest AIC (Akaike Information Criterion ).
best_aic = np.inf
best_pdq = None
best_seasonal_pdq = None
tmp_model = None
best_mdl = None
# Iteration Process
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            tmp_mdl = sm.tsa.statespace.SARIMAX(series[-1000:],
                                                order = param,
                                                seasonal_order = param_seasonal,
                                                enforce_stationarity=True,
                                                enforce_invertibility=True)
            res = tmp_mdl.fit()
            if res.aic < best_aic:
                best_aic = res.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
                best_mdl = tmp_mdl
        except:
            print("Unexpected error:", sys.exc_info()[0])
            continue
print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))

def sarimax(series,steps):    
    # define SARIMAX model and fit it to the data
    mdl = sm.tsa.statespace.SARIMAX(series,
                                    order=(2, 1, 0),
                                    seasonal_order=(1, 0, 0, 12),
                                    enforce_stationarity=True,
                                    enforce_invertibility=True)
    res = mdl.fit()
    pred_uc = res.get_forecast(steps=780)
    y_hat = pred_uc.predicted_mean

pyplot.plot(validate.values)
pyplot.plot(y_hat[:720], color='red')
pyplot.show()


pyplot.plot(validate.values)
pyplot.plot(aaa[:720], color='red')
pyplot.show()