import pandas as pd
import numpy as np

df_alert = pd.read_excel('Test.xlsx', sheet_name='Sheet1', index=True)
df_work = df_alert
df_temp = pd.read_excel('PredictBatteryIndicator.xlsx', sheet_name='Sheet1', index=True)
df_work['배터리량'] = df_temp['real_배터리량_Total']
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
        # print(date_new)

    if date_new == '2014-12-31 23:00:00':
        df_revised = df_work[:index+2]
        df_revised["배터리량"] = df_revised["배터리량"].astype('int')
        print("hi")
        df_revised["날짜"][8759] = '2015-01-01 24:00:00'
        break

df_revised["시간"] = df_revised["배터리량"]
for index, row in df_revised.iterrows():
    temp_per = df_revised["배터리량"][index]
    temp_t = int(temp_per) * 20
    df_revised["시간"][index] = temp_t

timeAVG = df_revised["시간"].to_numpy()
time2014 = np.reshape(timeAVG, (365, -1))
meanTime2014 = np.mean(time2014, axis=1)
aa = df_revised["날짜"]
#aa['평균'] = meanTime2014

meanTimeq = meanTime2014.tolist()
dt_index = pd.date_range("20140101", "20141201", freq="MS")
dt_list = dt_index.strftime("%Y-%m").tolist()
date = pd.DataFrame(meanTimeq)
dff = pd.DataFrame(dt_list)
dff["평균"] = date
dff["평균"]= dff["평균"].astype(int)

print(dff)

#print(time2014)