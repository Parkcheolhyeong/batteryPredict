import pandas as pd
import numpy

df = pd.read_excel("PredictBatteryIndicator.xlsx", sheet_name="Sheet1", index=False)
batteryReal = df["real_배터리량_Total"].to_numpy()

lstbtr = batteryReal.tolist()

dff = pd.DataFrame(lstbtr[:8710])
lstbtr = lstbtr[:8710]
#list(map(str, lstbtr))
list(map(int, lstbtr))
print(lstbtr)