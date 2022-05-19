# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2022-05-17T13:50:57.262448Z","iopub.execute_input":"2022-05-17T13:50:57.263447Z","iopub.status.idle":"2022-05-17T13:50:57.510992Z","shell.execute_reply.started":"2022-05-17T13:50:57.263385Z","shell.execute_reply":"2022-05-17T13:50:57.509961Z"}}
import numpy as np
import pandas as pd

df = pd.DataFrame(columns = ['year', 'month', 'day', 'weekday', 'holiday', 'long_holiday', 'holiday_seq', 'holiday_back_seq'])

#df.columns = ['year', 'month', 'day', 'weekday', 'holiday', 'long_holiday', 'holiday_seq', 'holiday_back_seq']
month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30 ,31]
month_2016_days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30 ,31]


for i in range(365):
    df.loc[i, 'year'] = 2015
    weekday = (i+3)%7 + 1
    df.loc[i, 'weekday'] = weekday
    if weekday == 6 or weekday == 7:
        df.loc[i, 'holiday'] = 1

now_month = 0
cnt = 0
for month_day in month_days:
    now_month += 1
    for m in range(month_day):
        df.loc[cnt, 'month'] = now_month
        df.loc[cnt, 'day'] = m+1
        cnt += 1
        
        
#1:1,2,3,4
#2:18-23,27,28 +30
#3:1, +58
#4:3-6 +89
#5:1,2,3 +119
#6:19,20,21
#9:26,27,28
#10:9,10,11

long_hol = [0, 1, 2, 3, 48, 49, 50, 51, 52, 53, 57, 58, 59, 92, 93, 94, 95, 
            120, 121, 122, 169, 170, 171, 268, 269, 270, 281, 282, 283]

group_long_hol = [[0, 1, 2, 3], 
                  [48, 49, 50, 51, 52, 53], 
                  [57, 58, 59], 
                  [92, 93, 94, 95], 
                  [120, 121, 122], 
                  [169, 170, 171], 
                  [268, 269, 270], 
                  [281, 282, 283]]

df.loc[long_hol, 'long_holiday'] = 1
df.loc[long_hol, 'holiday'] = 1
for group in group_long_hol:
    start = group[0]
    end = group[-1]
    for index in group:
        df.loc[index, 'holiday_seq'] = index-start+1
        df.loc[index, 'holiday_back_seq'] = end-index+1

df = df.fillna(0)
print(df)
df.to_csv('calendar.csv', index=None)

# %% [code] {"execution":{"iopub.status.busy":"2022-05-17T13:50:44.808127Z","iopub.execute_input":"2022-05-17T13:50:44.808411Z","iopub.status.idle":"2022-05-17T13:50:45.140548Z","shell.execute_reply.started":"2022-05-17T13:50:44.808382Z","shell.execute_reply":"2022-05-17T13:50:45.139542Z"}}
df_2016 = pd.DataFrame(columns = ['year', 'month', 'day', 'weekday', 'holiday', 'long_holiday', 'holiday_seq', 'holiday_back_seq'])

#df.columns = ['year', 'month', 'day', 'weekday', 'holiday', 'long_holiday', 'holiday_seq', 'holiday_back_seq']
#month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30 ,31]
month_2016_days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30 ,31, 31, 28, 31, 28] #April

for i in range(484):
    if i < 366:
        df_2016.loc[i, 'year'] = 2016
    else:
        df_2016.loc[i, 'year'] = 2017
    weekday = (i+3)%7 + 1
    df_2016.loc[i, 'weekday'] = weekday
    if weekday == 6 or weekday == 7:
        df_2016.loc[i, 'holiday'] = 1
        
now_month = 0
cnt = 0
for month_day in month_2016_days:
    now_month += 1
    if now_month > 12: now_month = 1
    for m in range(month_day):
        df_2016.loc[cnt, 'month'] = now_month
        df_2016.loc[cnt, 'day'] = m+1
        cnt += 1
        
long_hol = [0, 1, 2, 36, 37, 38, 39, 40, 41, 42, 43, 44, 57, 58, 59, 92, 93, 94, 95, 
            160, 161, 162, 163, 258, 259, 260, 261, 281, 282, 283, 365, 366, 367, 
            392, 393, 394, 395, 421, 422, 423, 424, 456, 457, 458, 459]

group_long_hol = [[0, 1, 2]
                  , [36, 37, 38, 39, 40, 41, 42, 43, 44]
                  , [57, 58, 59]
                  , [92, 93, 94, 95]
                  , [160, 161, 162, 163]
                  , [258, 259, 260, 261]
                  , [281, 282, 283]
                  , [365, 366, 367]
                  , [392, 393, 394, 395]
                  , [421, 422, 423, 424]
                  , [456, 457, 458, 459]]


df_2016.loc[long_hol, 'long_holiday'] = 1
df_2016.loc[long_hol, 'holiday'] = 1
for group in group_long_hol:
    start = group[0]
    end = group[-1]
    for index in group:
        df_2016.loc[index, 'holiday_seq'] = index-start+1
        df_2016.loc[index, 'holiday_back_seq'] = end-index+1

df_2016 = df_2016.fillna(0)


print(df_2016)

# %% [code] {"execution":{"iopub.status.busy":"2022-05-17T13:53:45.518308Z","iopub.execute_input":"2022-05-17T13:53:45.51965Z","iopub.status.idle":"2022-05-17T13:53:45.538538Z","shell.execute_reply.started":"2022-05-17T13:53:45.519577Z","shell.execute_reply":"2022-05-17T13:53:45.53699Z"}}
df_all = pd.concat([df, df_2016])
df_all = df_all.reset_index(drop=True)
print(df_all)
df_all.to_csv('calendar_final.csv', index=None)