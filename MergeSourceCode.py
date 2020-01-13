import sys
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import QAbstractTableModel, Qt
import tkinter
from tkinter import messagebox
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


df = pd.read_excel('Enercamp1001.xlsx', sheet_name='Total', index=True)
a_lst = []

#UI파일 연결
#단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
form_class = uic.loadUiType("pushbuttonTest.ui")[0]
class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None

#class comparePreviousValue():
#    for()

#class ComparePreviousCell:
#    def __init__(self):
#        self.data = df

#    def method():



#화면을 띄우는데 사용되는 Class 선언
class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)

        #버튼에 기능을 연결하는 코드
        self.btn_run.clicked.connect(self.btnRunFunction)
        self.btn_exit.clicked.connect(self.btnExitFunction)
        self.btn_load.clicked.connect(self.btnLoadFunction)

    #btn_1이 눌리면 작동할 함수
    def btnRunFunction(self) :
        #print("btn_1 Clicked")
        lstmprd = LSTMPredictor()
        lstmprd.run()

        count = 0
        # self.tableWidget.setColumnCount(5)
        df = pd.read_excel('Test.xlsx', sheet_name='Sheet1', index=True)
        df_work = df
        df_temp = pd.read_excel('PredictBatteryIndicator.xlsx', sheet_name='Sheet1', index=True)
        df_work['배터리량'] = df_temp['배터리량']
        ColumList = ['날짜', '배터리량', 'Alert']
        df_work = df_work[ColumList]

        date_old = df_work['날짜'][len(df_work)-1]

        date_split = date_old.split('-')
        year_old = str(date_split[0])
        year_new = str(int(date_split[0]) + 1)

        for index, row in df_work.iterrows():
            if (index < len(df_work)):
                date_old = df_work['날짜'][index]
                date_split = date_old.split('-')
                year_old = str(date_split[0])

                date_split[0] = year_new
                date_new = '-'.join(date_split)
                df_work['날짜'][index] = date_new

        model = pandasModel(df_work)

        dailyAvg(self)
        self.tableView_2.setModel(model)
        self.tableView_2.resizeColumnsToContents()



    #btn_2가 눌리면 작동할 함수
    def btnExitFunction(self) :
        exit()

    # btn_3가 눌리면 작동할 함수
    def btnLoadFunction(self):
        header_labels = ['Column 1', 'Column 2', 'Column 3', 'Column 4', 'Column 5', 'Column 6']
        #print("btn_3 Clicked")
        model = pandasModel(df)

        count = 0
        #self.tableWidget.setColumnCount(5)


        self.tableView.setModel(model)
        for index, row in df.iterrows():
            if (index < len(df) - 1):
                if abs(int(df['배터리량'][index].item()) - int(df['배터리량'][index + 1].item())) > 40:
                    a_lst.append(1)
                    count = count +1
                else:
                    a_lst.append(0)

        a_lst.append(0)
        df["Alert"] = a_lst

        # data = df.assign(flag=df['배터리량'].gt(df['배터리량'].shift()))
        # do something
        df['Alert'] = (df['Alert'] == 1)
        alertValue = df[['날짜', 'Alert']]
        batteryAvg = df['배터리량']


        #print(df)
        alertLen = len(df['Alert'] == True)
        #print(alertValue)
        model = pandasModel(alertValue)

        # 1년에 1~8544할하루 365일 하루 24시간



        self.tableView_3.setModel(model)
        self.tableView.resizeColumnsToContents()
        self.tableView_3.resizeColumnsToContents()
        messageBox(count)

        alertValue.to_excel('Test.xlsx')
        # read csv file
        #df = pd.read_excel("./JuO_temp.xlsx")#, names=['date', 'Percent', 'Volt', 'Charge'])  # df is pandas.DataFrame
        #print("##### data #####")
        #print(df)




class LSTMPredictor() :
    global select_predict
    global select_day
    global select_count

    def __init__(self):
        tf.debugging.set_log_device_placement(True)
        # mirrored_strategy =  tf.distribute.MirroredStrategy()
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
        # default setting

    def MAPE(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # convert series to supervised learning
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def run(self):
        self.select_predict = "배터리량"
        self.select_day = "Total"
        self.select_count = "5"
        print("data : " + self.select_predict)
        print("select day : " + self.select_day)
        print("select year : " + self.select_count)

        data = pd.read_excel('Enercamp.xlsx', sheet_name=self.select_day)
        df = pd.DataFrame(data, columns=['배터리량', '전기량', '충전량'])
        # df = pd.DataFrame(data, columns = ['ex_load', 'temp', 'rainfall', 'wind', 'humidity', 'cloud', 'discomfor_index', 'wind_temp'] )
        # df["smp"] = data.target
        X = df.drop(self.select_predict, 1)  # Feature Matrix
        y = df[self.select_predict]  # Target Variable
        # X = df.drop("ex_load",1)   #Feature Matrix
        # y = df["ex_load"]          #Target Variable
        df.head()

        plt.figure(figsize=(20, 17))
        cor = df.corr()
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        # plt.show()
        # plt.savefig(self.select_predict + '_'+ select_day + '.png')

        # Correlation with output variable
        cor_target = abs(cor[self.select_predict])
        # cor_target = abs(cor["ex_load"])
        # Selecting highly correlated features
        relevant_features = cor_target[cor_target > 0.5]
        print(df.columns)
        print(cor_target)
        print("total features : " + str(len(cor_target)))
        print(relevant_features.index)
        print(relevant_features)
        print("input features : " + str(len(relevant_features)))

        # relevant_features = 12

        # load dataset
        dataset = pd.read_excel('Enercamp.xlsx', sheet_name=self.select_day)
        # print(dataset)
        # dataset = pd.DataFrame(dataset, columns = ['smp', 'coal', 'solar', 'wind', 'hydraulic', 'ocean', 'bio', 'LNG', 'wind', 'nuclear', 'b_coal', 'gas'] )
        dataset = pd.DataFrame(dataset, columns=relevant_features.index)

        values = dataset.values
        # integer encode direction
        # encoder = LabelEncoder()
        # values[:,1] = encoder.fit_transform(values[:,1])
        # ensure all data is float
        values = values.astype('float32')
        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        # specify the number of lag hours
        n_hours = 3
        n_features = len(relevant_features)
        # n_features = 12
        # calculate data
        tmp = -len(relevant_features) + 1
        # tmp = -relevant_features + 1
        # frame as supervised learning
        reframed = self.series_to_supervised(scaled, n_hours, 1)
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
        history = model.fit(train_X, train_y, epochs=1, batch_size=24, validation_data=(test_X, test_y), verbose=2,
                            shuffle=False)
        # plot history
        pyplot.plot(history.history['loss'], label='train')
        # pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        # pyplot.show()

        # make a prediction
        yhat = model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))
        # invert scaling for forecast
        inv_yhat = concatenate((yhat, test_X[:, tmp:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, 0]
        # invert scaling for actual
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = concatenate((test_y, test_X[:, tmp:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, 0]
        # calculate RMSE
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        #print('Test RMSE: %.3f' % rmse)
        # calculate MAPE
        mape = self.MAPE(inv_y, inv_yhat)
        print('Test MAPE: %.3f' % mape)

        # print(inv_y)
        # for i in range(120):
        #    print(inv_y[i])
        # print(len(inv_y))
        # print(inv_yhat)
        # for i in range(120):
        #    print(inv_yhat[i])
        # print(len(inv_yhat))

        for i in range(len(inv_yhat)):
            dataset.loc[i:len(inv_yhat), 'predict_' + self.select_predict + '_' + self.select_day] = inv_yhat[i]
            dataset.loc[i:len(inv_y), 'real_' + self.select_predict + '_' + self.select_day] = inv_y[i]

        # dataset.to_excel(select_predict + 'Battery' + '_' + self.select_day + '.xlsx')
        dataset.to_excel('PredictBatteryIndicator.xlsx')






def messageBox(i):
    root = tkinter.Tk()
    root.withdraw()
    msg = "과다사용 횟수: " + str(i)
    messagebox.showinfo(title="Alert Notification", message=msg)

def dailyAvg(self):
    timeAVG = df["시간"].to_numpy()
    daily2009 = timeAVG[:8760]
    daily2010 = timeAVG[8760:17520]
    daily2011 = timeAVG[17520:26280]
    daily2012 = timeAVG[26280:35064]
    daily2013 = timeAVG[35064:43824]
    time2009 = np.reshape(daily2009, (365, -1))
    time2010 = np.reshape(daily2010, (365, -1))
    time2011 = np.reshape(daily2011, (365, -1))
    time2012 = np.reshape(daily2012, (366, -1))
    time2013 = np.reshape(daily2013, (365, -1))

    meanTime2009 = np.mean(time2009, axis=1)
    meanTime2010 = np.mean(time2010, axis=1)
    meanTime2011 = np.mean(time2011, axis=1)
    meanTime2012 = np.mean(time2012, axis=1)

    meanTime = meanTime2009.tolist() + meanTime2010.tolist() + meanTime2011.tolist() + meanTime2012.tolist()

    minValue = min(meanTime)
    dt_index = pd.date_range("20090101", "20121201", freq="D")
    dt_list = dt_index.strftime("%Y-%m-%d").tolist()
    date = pd.DataFrame(meanTime)
    dff = pd.DataFrame(dt_list)
    dff["평균"] = date

    model = pandasModel(dff)

    self.tableView_4.setModel(model)
    self.tableView_4.resizeColumnsToContents()


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()