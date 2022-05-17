# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 13:48:45 2016

@author: mfchang
"""
import os

import pandas as pd

from datetime import datetime
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier

#os.chdir('D:/TIP/TT')
TTDir = 'M06TT/'

day1 = datetime.strptime('20150101', "%Y%m%d")
day1_str = day1.strftime("%Y%m%d %H%M%S")
dayLast = datetime.strptime('20170428 235959', "%Y%m%d %H%M%S")
dayLast_str = dayLast.strftime("%Y%m%d %H%M%S")

train_dayLast = datetime.strptime('20170331 235959', "%Y%m%d %H%M%S")

test_dayFirst = train_dayLast + timedelta(seconds = 1)

#pathid = 1
pathids = [1,3,4,6,8,9,12,15,16,17,18,19]

for pathid in pathids:
    print("now pathid: ", pathid)
    filename = TTDir + 'Path' + repr(pathid) + '_20150101TTFile2.csv'
    pathTT_df  = pd.read_csv(filename, sep = '\t', infer_datetime_format = True, index_col = 0)

    pathTT_df.index = pd.to_datetime(pathTT_df.index)
    pathTT_df['TT'] = pathTT_df['TT'].astype(int)

    dt_names = ['dt', 'year', 'month', 'holiday', 'hldy_seq', 'weekday', 'timeslot']

    dt_dtypes = {'year': int, 'month': int, 'holiday': int, 'hldy_seq': int, 'weekday': int, 'timeslot': int}

    dt_df = pd.read_csv('static_files/weekday.txt', sep = '\t', names = dt_names, dtype = dt_dtypes, infer_datetime_format = True, index_col = 0)

    dt_df.index = pd.to_datetime(dt_df.index)

    sel_dt_df_index = pd.date_range(day1, periods=(dayLast - day1).total_seconds() // 300 + 1, freq='5T')

    sel_dt_df = dt_df.loc[sel_dt_df_index, :]
    sel_pathTT_df = pathTT_df.loc[sel_dt_df_index, :]

    train_dt_index = pd.date_range(day1+timedelta(minutes = 5), periods=(train_dayLast - day1).total_seconds() // 300 + 1, freq='5T')

    test_dt_index = pd.date_range(test_dayFirst, periods=(dayLast - test_dayFirst).total_seconds() // 300 + 1, freq='5T')


    mrg_df = pd.concat([sel_pathTT_df, sel_dt_df], axis = 1)

    calcMean_df = mrg_df.loc[train_dt_index, :]

    TT_mean_df = calcMean_df[['TT', 'hldy_seq', 'weekday', 'timeslot']].groupby(['hldy_seq', 'weekday', 'timeslot']).mean()

    TT_mean_dict = TT_mean_df.to_dict()['TT']

    holi_mean_df = calcMean_df[['TT', 'hldy_seq', 'timeslot']].groupby(['hldy_seq', 'timeslot']).mean()

    holi_TT_mean_dict = holi_mean_df.to_dict()['TT']


    mrg_df['TT_mean'] = np.nan

    for index, row in mrg_df.iterrows():
        if mrg_df.loc[index, 'holiday'] == 0:
            mrg_df.loc[index, 'TT_mean'] = TT_mean_dict[(row['hldy_seq'], row['weekday'], row['timeslot'])]
        else:
            mrg_df.loc[index, 'TT_mean'] = holi_TT_mean_dict[(row['hldy_seq'], row['timeslot'])]

    #long-term prediction
    trainData = mrg_df.loc[train_dt_index, ['year', 'month', 'holiday', 'hldy_seq',
                            'weekday', 'timeslot', 'TT_mean']].values

    trainTarget = mrg_df.loc[train_dt_index, 'TT'].values

    #long-term prediction
    testData = mrg_df.loc[test_dt_index, ['year', 'month', 'holiday', 'hldy_seq',
                            'weekday', 'timeslot', 'TT_mean']].values

    testTarget = mrg_df.loc[test_dt_index, 'TT'].values

    print(trainData.shape)
    print(testData.shape)

    # haha
    #trainData.("train_data.dat")
    #trainTarget.tofile("train_target.dat")
    #testData.tofile("test_data.dat")
    #testTarget.tofile("test_target.dat")

    np.save('array_files/train_data_%d.npy'%pathid, trainData)
    np.save('array_files/train_target_%d.npy'%pathid, trainTarget)
    np.save('array_files/test_data_%d.npy'%pathid, testData)
    np.save('array_files/test_target_%d.npy'%pathid, testTarget)

    continue
    #print(trainData.dtype)

    #df_train_data = pd.DataFrame(trainData)
    #df_train_target = pd.DataFrame(trainTarget)
    #df_test_data = pd.DataFrame(testData)
    #df_test_target = pd.DataFrame(testTarget)
    
    #df_train_data.to_csv("train_data.csv")
    #df_train_target.to_csv("train_target.csv")
    #df_test_data.to_csv("test_data.csv")
    #df_test_target.to_csv("test_target.csv")

    #trainData = np.load("array_files/train_data_%d.npy"%pathid)
    #trainTarget = np.load("array_files/train_target_%d.npy"%pathid)
    #testData = np.load("array_files/test_data_%d.npy"%pathid)
    #testTarget = np.load("array_files/test_target_%d.npy"%pathid)
    #haha
    alpha = 0.48

    n_estimators_s = [50, 100, 200, 300]
    max_depth_s = [5, 7, 9, 11]
    loss_s = ['quantile']
    min_samples_leaf_s = [50, 100,150]
    min_samples_split_s = [50, 100, 150]
    max_features_s = ['auto', 'sqrt']

    #n_estimators_s = [50, 100]
    #max_depth_s = [5]
    #loss_s = ['quantile']
    #min_samples_leaf_s = [50]
    #min_samples_split_s = [50]
    #max_features_s = ['auto', 'sqrt']

    mape_compared = 1
    mse_compared = 1
    best_model = ""
    final_pred = ""

    for n_estimators in n_estimators_s:
        for max_depth in max_depth_s:
            for loss in loss_s:
                for min_samples_leaf in min_samples_leaf_s:
                    for min_samples_split in min_samples_split_s:
                        for max_features in max_features_s:

                            clf = GradientBoostingRegressor(loss=loss, alpha=alpha,
                                                            n_estimators=n_estimators, max_depth=max_depth,
                                                            learning_rate=.03, min_samples_leaf=min_samples_leaf,
                                                            min_samples_split=min_samples_split,
                                                            max_features=max_features, random_state=187)

                            #clf = RandomForestClassifier(random_state=187)

                            clf.fit(trainData, trainTarget)

                            testPrdct = clf.predict(testData)

                            mse = ((testTarget - testPrdct) ** 2).mean()

                            mape = (abs((testTarget - testPrdct)) / testTarget).mean()

                            if mape < mape_compared :
                                mape_compared = mape
                                mse_compared = mse
                                best_model = clf
                                final_pred = testPrdct

    plt.figure(num=None, figsize=(168, 6))

    plt.plot(testTarget)

    plt.plot(final_pred)

    plt.savefig(TTDir + 'Path_' + repr(pathid) + '_' + day1_str + '_' + dayLast_str + '.png')

    np.savetxt(TTDir + 'Path_' + repr(pathid) + '_' + day1_str + '_' + dayLast_str + '.txt',
               np.transpose(np.stack((testTarget, testPrdct))), delimiter="\t")

    with open('grid_search/result_%d.txt' % pathid, 'w') as f:
        f.write(str(mape_compared) + '\n')
        f.write(str(mse_compared) + '\n')
        f.write(str(best_model) + '\n')

    #print(mape, mse)




