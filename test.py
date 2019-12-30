import pandas as pd

df = pd.read_excel('Enercamp_1.xlsx', sheet_name='Sheet1', index=True)

print(len(df))
a_lst = []
for index, row in df.iterrows():
    if (index < len(df)-1):
        if int(df['배터리량'][index].item()) < int(df['배터리량'][index + 1].item()):
            a_lst.append(1)
        else:
            a_lst.append(2)

a_lst.append(0)
df["Alert"] = a_lst
print(df)
#print(df)