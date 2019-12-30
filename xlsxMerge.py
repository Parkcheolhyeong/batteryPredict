import pandas as pd
import datetime
df = pd.read_excel('Test.xlsx', sheet_name='Sheet1', index=True)
df_temp = pd.read_excel('PredictBatteryIndicator.xlsx', sheet_name='Sheet1', index=True)
df['배터리량'] = df_temp['배터리량']

ColumList = ['날짜', '배터리량', 'Alert']
df = df[ColumList]
#print(df)

date_old = df['날짜'][len(df)-1]
#print(myDatetime.replace("2009", "2020"))

date_split = date_old.split('-')
year_old = str(date_split[0])
year_new = str(int(date_split[0])+1)
date_split[0] = year_new
#a.insert(year_old, year_new)
print(date_split)

date_new = '-'.join(date_split)
print(date_new)