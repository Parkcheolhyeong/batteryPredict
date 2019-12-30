import sys
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import QAbstractTableModel, Qt
import tkinter

from tkinter import messagebox

df = pd.read_excel('Enercamp_1.xlsx', sheet_name='Total', index=True)
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
        print("btn_1 Clicked")

    #btn_2가 눌리면 작동할 함수
    def btnExitFunction(self) :
        exit()

    # btn_3가 눌리면 작동할 함수
    def btnLoadFunction(self):
        header_labels = ['Column 1', 'Column 2', 'Column 3', 'Column 4']
        print("btn_3 Clicked")
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

        #print(df)
        alertLen = len(df['Alert'] == True)
        print(alertValue)
        model = pandasModel(alertValue)
        self.tableView_3.setModel(model)
        messageBox(count)
        # read csv file
        #df = pd.read_excel("./JuO_temp.xlsx")#, names=['date', 'Percent', 'Volt', 'Charge'])  # df is pandas.DataFrame
        print("##### data #####")
        #print(df)

def messageBox(i):
    root = tkinter.Tk()
    root.withdraw()
    messagebox.showinfo(title="Hi", message=i)


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()