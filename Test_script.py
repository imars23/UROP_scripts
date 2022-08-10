import pandas as pd

path = r'/home/imars23/Desktop/Data/'

data = pd.read_excel (path + r'Pumpout_Shots_Database.xlsx', engine = 'openpyxl', usecols = 'A:U', nrows = 52 )
data_list = data.values.tolist()

print(data_list)
