import sys
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5 import uic

#UI파일 연결
#단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
form_class = uic.loadUiType("pushbuttonTest.ui")[0]

#화면을 띄우는데 사용되는 Class 선언
class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)

        #버튼에 기능을 연결하는 코드
        self.btn_run.clicked.connect(self.button1Function)
        self.btn_exit.clicked.connect(self.button2Function)
        self.btn_load.clicked.connect(self.button3Function)



    #btn_1이 눌리면 작동할 함수
    def button1Function(self) :
        print("btn_1 Clicked")

    #btn_2가 눌리면 작동할 함수
    def button2Function(self) :
        print("btn_2 Clicked")
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setRowCount(3)

        for r in 3:
            for c in 3:
                self.tableWidget.setItem(0, 0, QTableWidgetItem("000020"))
            self.table.wid
        self.tableWidget.setItem(0, 0, QTableWidgetItem("000020"))
        self.tableWidget.setItem(0, 1, QTableWidgetItem("삼성전자"))
        self.tableWidget.setItem(1, 0, QTableWidgetItem("000030"))
        self.tableWidget.setItem(1, 1, QTableWidgetItem("현대차"))
        self.tableWidget.setItem(2, 0, QTableWidgetItem("000080"))

    # btn_3가 눌리면 작동할 함수
    def button3Function(self):
        print("btn_3 Clicked")
        # read csv file
        df = pd.read_excel("./JuO_temp.xlsx")#, names=['date', 'Percent', 'Volt', 'Charge'])  # df is pandas.DataFrame
        print("##### data #####")
        print(df)




if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()