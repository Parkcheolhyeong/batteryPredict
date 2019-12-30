from openpyxl import load_workbook

# data_only=Ture로 해줘야 수식이 아닌 값으로 받아온다.
load_wb = load_workbook("K-WATER_total.xlsx", data_only=True)
# 시트 이름으로 불러오기
load_ws = load_wb['Total']

for row in load_ws.rows:
    print(row)