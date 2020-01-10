import pandas as pd
import numpy

df = pd.read_excel("Enercamp1001.xlsx", sheet_name="Sheet1", index=False)
timeAVG = df["시간"].to_numpy()

daily2009 = timeAVG[:8760]
daily2010 = timeAVG[8760:17520]
daily2011 = timeAVG[17520:26280]
daily2012 = timeAVG[26280:35064]
daily2013 = timeAVG[35064:43824]
time2009 = numpy.reshape(daily2009, (365, -1))
time2010 = numpy.reshape(daily2010, (365, -1))
time2011 = numpy.reshape(daily2011, (365, -1))
time2012 = numpy.reshape(daily2012, (366, -1))
time2013 = numpy.reshape(daily2013, (365, -1))

meanTime2009 = numpy.mean(time2009, axis=1)
meanTime2010 = numpy.mean(time2010, axis=1)
meanTime2011 = numpy.mean(time2011, axis=1)
meanTime2012 = numpy.mean(time2012, axis=1)

meanTime = meanTime2009.tolist() + meanTime2010.tolist() +meanTime2011.tolist() +meanTime2012.tolist()
#print(meanTime2009)
    #print(len(meanTime2009))


df = pd.read_excel("Enercamp.xlsx", sheet_name="Total", index=False)
df_work = df
df_temp = pd.read_excel('PredictBatteryIndicator.xlsx', sheet_name='Sheet1', index=True)
df_work['배터리량'] = df_temp['배터리량']
ColumList = ['날짜']
df_work = df_work[ColumList]

date_old = df_work['날짜'][len(df_work) - 1]

date_split = date_old.split('-')
year_old = str(date_split[0])
year_new = str(int(date_split[0]) + 1)

for index, row in df_work.iterrows():
    if (index < len(df_work)):
        date_old = df_work['날짜'][index]
        date_split = date_old.split(' ')
        df_work['날짜'][index] = date_split[0]

#print(df_work)
dt_index = pd.date_range("20090101", "20121201", freq= "D")
# pandas.date_range(start='20160901', end='20161031',freq='W-MON')
# 을 하면 해당 기간 매주 월요일들만 추출합니다.

# type(dt_index) => DatetimeIndex
# DatetimeIndex => list(str)

minValue = min(meanTime)
dt_list = dt_index.strftime("%Y-%m-%d").tolist()
date = pd.DataFrame(meanTime)
dff = pd.DataFrame(dt_list)

dff["평균"] = date
print(dff)
print("최솟 값: " + str(int(minValue)) +"분")