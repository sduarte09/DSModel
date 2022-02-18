
### Model Keras Version 1
from keras.models import Sequential
from keras.layers import Dense
import ML_Functions.Chaos_Statistics

# fix random seed for reproducibility
lag,dim,lyapunov,caos=ML_Functions.Chaos_Statistics.chaos_analysis(series)
X=series[:-1]
Y = series[1:]

# Create model
model = Sequential()
model.add(Dense(20, input_dim=1, init='Orthogonal', activation='relu'))
model.add(Dense(1,init='Orthogonal', activation='relu'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=10, batch_size=500,verbose=2)
# calculate predictions
for i in range(steps):  
  act=[series[-1]]
  fut=model.predict(act)
  series=np.append(series,fut)
predict_2=series[-steps:]



if len(data)>20000:
    data=data[-20000:] 
series=data.values   
    
## fix random seed for reproducibility
#seed =int(np.nanmean(series))
#np.random.seed(seed)
#lag=ML_Functions.Chaos_Statistics.delay(series)
#x1=series[:-lag]
#x2=series[lag-1:-1]
#X=np.transpose(np.vstack((x1, x2)))
#Y = series[lag:]
## create model
#model = Sequential()
#model.add(Dense(64, input_dim=2, init='uniform', activation='relu'))
#model.add(Dense(1, input_dim=2, init='uniform', activation='relu'))
## Compile model
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
## Fit the model
#model.fit(X, Y, epochs=100, batch_size=steps,verbose=2)
## calculate predictions
#for i in range(steps):  
#  act=np.array([series[-lag],series[-1]]).reshape(1,2)
#  fut=model.predict(act)
#  series=np.append(series,fut)
#predict_2=series[-steps:]