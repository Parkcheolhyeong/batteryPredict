from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
import tensorflow as tf


tf.debugging.set_log_device_placement(True)
#mirrored_strategy =  tf.distribute.MirroredStrategy()
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])


#default setting
select_predict = "smp"
select_day = "weekday"
select_count = "0"

def func_select_predict():
    global select_predict
    select_number = input("1. load \n2. smp \n3. rank \ninput : ")
    if select_number == '1':
        select_predict = "load"
    elif select_number == '2':
        select_predict = "smp"
    elif select_number == '3':
        select_predict = "rank"

def func_select_day():
    global select_day
    select_number = input("0. Total \n1. weekday \n2. weekend \n3. holiday \ninput : ")
    if select_number == '0':
        select_day = "Total"
    elif select_number == '1':
        select_day = "weekday"
    elif select_number == '2':
        select_day = "weekend"
    elif select_number == '3':
        select_day = "holiday"
    elif select_number == '4':
        select_day = "weekday_2years"

def func_select_count():
    global select_count
    select_count = input("select years number : ")

#func_select_predict()
#func_select_day()
#func_select_count()

select_predict = "load"
select_day = "Total"
select_count = "5"
print("data : " + select_predict)
print("select day : " + select_day)
print("select year : " + select_count)

data = pd.read_excel('battery_total.xlsx', sheet_name = select_day )
df = pd.DataFrame(data, columns = ['load', 'rank', 'fuel_cell'] )
#df = pd.DataFrame(data, columns = ['ex_load', 'temp', 'rainfall', 'wind', 'humidity', 'cloud', 'discomfor_index', 'wind_temp'] )


#df["smp"] = data.target
X = df.drop(select_predict,1)   #Feature Matrix
y = df[select_predict]          #Target Variable
#X = df.drop("ex_load",1)   #Feature Matrix
#y = df["ex_load"]          #Target Variable
df.head()

plt.figure(figsize=(20,17))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
#plt.show()
#plt.savefig(select_predict + '_'+ select_day + '.png')


#Correlation with output variable
cor_target = abs(cor[select_predict])
#cor_target = abs(cor["ex_load"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]

print(df.columns)
print(cor_target)
print("total features : " + str(len(cor_target)))
print(relevant_features.index)
print(relevant_features)
print("input features : " + str(len(relevant_features)))



def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
   n_vars = 1 if type(data) is list else data.shape[1]
   df = DataFrame(data)
   cols, names = list(), list()
   # input sequence (t-n, ... t-1)
   for i in range(n_in, 0, -1):
      cols.append(df.shift(i))
      names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
   # forecast sequence (t, t+1, ... t+n)
   for i in range(0, n_out):
      cols.append(df.shift(-i))
      if i == 0:
         names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
      else:
         names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
   # put it all together
   agg = concat(cols, axis=1)
   agg.columns = names
   # drop rows with NaN values
   if dropnan:
      agg.dropna(inplace=True)
   return agg

#relevant_features = 12

# load dataset
dataset = pd.read_excel('battery_total.xlsx', sheet_name = select_day)
#print(dataset)
#dataset = pd.DataFrame(dataset, columns = ['smp', 'coal', 'solar', 'wind', 'hydraulic', 'ocean', 'bio', 'LNG', 'wind', 'nuclear', 'b_coal', 'gas'] )
dataset = pd.DataFrame(dataset, columns = relevant_features.index )

values = dataset.values
# integer encode direction
#encoder = LabelEncoder()
#values[:,1] = encoder.fit_transform(values[:,1])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# specify the number of lag hours
n_hours = 3
n_features = len(relevant_features)
#n_features = 12
#calculate data
tmp = -len(relevant_features) + 1
#tmp = -relevant_features + 1
# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, 1)
print(reframed.shape)

# split into train and test sets
split_data = ((reframed.shape[0] - 1) / 5) / 24
print(int(round(split_data)))
values = reframed.values
n_train_hours = int(round(split_data)) * 24 * 4

train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
print("train : ", train)
print("test : ", test)
# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), test_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=5, batch_size=24, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
#pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, tmp:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, tmp:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
# calculate MAPE
mape = MAPE(inv_y, inv_yhat)
print('Test MAPE: %.3f' % mape)

#print(inv_y)
#for i in range(120):
#    print(inv_y[i])
#print(len(inv_y))
#print(inv_yhat)
#for i in range(120):
#    print(inv_yhat[i])
#print(len(inv_yhat))


for i in range(len(inv_yhat)):
    dataset.loc[i:len(inv_yhat), 'predict_' + select_predict + '_' + select_day] = inv_yhat[i]
    dataset.loc[i:len(inv_y), 'real_' + select_predict + '_' + select_day] = inv_y[i]

dataset.to_excel(select_predict + 'Battery'+ '_'+ select_day + '.xlsx')

