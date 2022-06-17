
# coding=utf-8
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

pd.set_option('mode.chained_assignment', None)


class Chebyrima:

    target_field = 'Temp'

    def __init__(self, filename):
        df = pd.read_csv(filename)
        self.data = df.copy()

    def indexParser(self, index='LocalTime', form='%Y-%m-%d %H:%M:%S'):
        print('time transform start')
        time = []
        for i in range(self.data.shape[0]):
            time.append(datetime.strptime(self.data.iloc[i][index], form))
        self.data['LocalTime'] = pd.Series(time)
        self.data = self.data.set_index('LocalTime')
        print('time transform done')

    def normalization(self):
        self.data = self.data.round(2)

    def timeParser(self):
        data = self.data['LocalTime'].copy()
        time = []
        for i in range(data.shape[0]):
            time_split = data.iloc[i].split(' ')
            date_split = time_split[0].split('/')
            year = date_split[0]
            month = date_split[1]
            day = date_split[2]
            sec = time_split[1]+":"+"00"
            today = year+'-'+month+'-'+day+' '+sec
            time.append(datetime.strptime(today, '%Y-%m-%d %H:%M:%S'))
        self.data['LocalTime'] = pd.Series(time)
        self.data = self.data.set_index('LocalTime')

    def newField(self):
        self.data[self.target_field + '_m'] = [np.nan] * self.data.shape[0]

    def outputToCsv(self, file, index=True):
        print("輸出檔案", file)
        self.data.to_csv(file, index=index)
        print("輸出檔案結束")

    def anomalyDetection(self, method='chebyshev', value_maximun=40, value_minimun=0):
        def timeParser(date, time, day=0):
            _date = (date + timedelta(days=day)).strftime('%Y-%m-%d')
            return datetime.strptime(_date + ' ' + time, '%Y-%m-%d %H:%M:%S')
        self.data[self.target_field][self.data[self.target_field]
                                     > value_maximun] = np.nan
        self.data[self.target_field][self.data[self.target_field]
                                     < value_minimun] = np.nan
        if method == 'chebyshev':
            start_day = self.data.index[0]
            end_day = self.data.index[-1]
            days = 0
            run = True
            while run:
                if days == 0:
                    start_time = start_day
                    end_time = timeParser(start_time.date(), '23:55:00')
                elif timeParser(start_day.date(), '00:00:00', day=days).date() == end_day.date():
                    start_time = timeParser(
                        start_day.date(), '00:00:00', day=days)
                    end_time = end_day
                    run = False
                else:
                    start_time = timeParser(
                        start_day.date(), '00:00:00', day=days)
                    end_time = timeParser(
                        start_day.date(), '23:55:00', day=days)
                avg = self.data.loc[start_time:end_time][self.target_field].mean(
                )
                std = self.data.loc[start_time:end_time][self.target_field].std(
                )
                std *= 2
                self.data.loc[start_time:end_time][self.target_field][self.data[self.target_field] > (
                    avg + std)] = np.nan
                self.data.loc[start_time:end_time][self.target_field][self.data[self.target_field] < (
                    avg - std)] = np.nan
                std = self.data.loc[start_time:end_time][self.target_field].std(
                )
                std *= 4
                self.data.loc[start_time:end_time][self.target_field][self.data[self.target_field] > (
                    avg + std)] = np.nan
                self.data.loc[start_time:end_time][self.target_field][self.data[self.target_field] < (
                    avg - std)] = np.nan
                self.condition = np.isnan(self.data[self.target_field])
                self.condition = list(self.condition)
                days += 1

    def reverse(self):
        self.data = self.data.reset_index()
        data_dict = self.data.to_dict(orient='list')
        data_dict_keys = list(data_dict.keys())
        data_dict_keys.pop(0)
        for i in data_dict_keys:
            data_dict[i].reverse()
        self.condition.reverse()
        self.data = pd.DataFrame(data_dict)
        self.data = self.data.set_index('LocalTime')

    def correctionVertical(self, window_size=24):
        mask = list(np.isnan(self.data[self.target_field]))
        pass_size = int(window_size / 2)
        data_size = self.data.shape[0]
        for i in range(len(mask)):
            if mask[i]:
                if i > window_size:
                    if self.condition[i - window_size:i].count(True) < pass_size:
                        if mask[i - window_size:i].count(True) == 0:
                            try:
                                fit1 = SARIMAX(self.data[self.target_field][i - window_size:i], seasonal_order=(
                                    0, 0, 0, 0), initialization='approximate_diffuse').fit()
                                predictions = fit1.predict(
                                    start=self.data.index[i], end=self.data.index[i], dynamic=True)
                                self.data[self.target_field][self.data.index[i]
                                                             ] = predictions.iloc[0]
                                mask[i] = False
                                self.data[self.target_field +
                                          '_m'][self.data.index[i]] = 'V'
                            except Exception as e:
                                print(e)
                                pass

    def correctionHorizontal(self, window_size=28):
        mask = list(np.isnan(self.data[self.target_field]))
        pass_size = int(window_size / 2)
        data_size = self.data.shape[0]
        c = 0
        for i in range(len(mask)):
            if mask[i]:
                if i > window_size * 288:
                    if self.condition[i - (window_size * 288):i:288].count(True) < pass_size:
                        if mask[i - (window_size * 288):i:288].count(True) == 0:
                            try:
                                fit1 = SARIMAX(self.data[self.target_field][i - (window_size * 288):i:288], seasonal_order=(
                                    0, 0, 0, 0), initialization='approximate_diffuse').fit()
                                predictions = fit1.predict(
                                    start=self.data.index[i], end=self.data.index[i], dynamic=True)
                                self.data[self.target_field][self.data.index[i]
                                                             ] = predictions[predictions.index[0]]
                                mask[i] = False
                                self.data[self.target_field +
                                          '_m'][self.data.index[i]] = 'H'
                            except Exception as e:
                                print(e)

    def smooth(self, gap=0.3, fix=48):
        a_array = self.data[self.target_field].copy().tolist()
        b_array = a_array.copy()
        b_array.pop()
        b_array.insert(0, 0)
        diff_value = np.array(a_array) - np.array(b_array)
        diff_array = pd.Series((diff_value).tolist())
        positive_mask = diff_array >= gap
        negative_mask = diff_array <= -gap
        for i in range(1, diff_array.shape[0]):
            if positive_mask[i] or negative_mask[i]:
                if diff_value[i] > 0:
                    n = diff_value[i] / 2
                    x = n / fix
                    for j in range(fix):
                        k = fix - j
                        try:
                            if not np.isnan(self.data.iloc[i+j][self.target_field]):
                                self.data[self.target_field][self.data.index[i + j]] -= x*k
                        except Exception as e:
                            pass
                        try:
                            if not np.isnan(self.data.iloc[i-j-1][self.target_field]):
                                self.data[self.target_field][self.data.index[i-j-1]] += x*k
                        except Exception as e:
                            pass
                elif diff_value[i] < 0:
                    n = -diff_value[i] / 2
                    x = n / fix
                    for j in range(fix):
                        k = fix - j
                        try:
                            if not np.isnan(self.data.iloc[i+j][self.target_field]):
                                self.data[self.target_field][self.data.index[i + j]] += x*k
                        except Exception as e:
                            pass
                        try:
                            if not np.isnan(self.data.iloc[i-j-1][self.target_field]):
                                self.data[self.target_field][self.data.index[i-j-1]] -= x*k
                        except Exception as e:
                            pass

    def gridSearch(self, data, maximun=2, minimun=0):
        p = d = q = range(minimun, maximun)
        pdq = list(itertools.product(p, d, q))
        print(pdq)
        aic_minimun = float('inf')
        order = (1, 1, 1)
        for param in pdq:
            try:
                aic = SARIMAX(data, order=param, seasonal_order=(
                    0, 0, 0, 0), enforce_stationarity=False, enforce_invertibility=False).fit().aic
                if aic < aic_minimun:
                    aic_minimun = aic
                    order = param
            except:
                continue
        return order


if __name__ == "__main__":
    file_name = str(input("請輸入站別名稱："))
    print("目前運行站別：{}".format(file_name))
    data = Chebyrima("{}.csv".format(file_name))
    data.timeParser()
    print("補遺開始")
    data.normalization()
    for i in ['Temp', 'Hum']:
        data.target_field = i
        if data.target_field == 'Temp':
            data.anomalyDetection()
        elif data.target_field == 'Hum':
            data.anomalyDetection(value_maximun=100, value_minimun=30)
    data.outputToCsv('./initdata/init_{}.csv'.format(file_name))
    for i in ['Temp', 'Hum']:
        data.target_field = i
        data.newField()
        data.correctionVertical()
        data.reverse()
        data.correctionVertical()
        data.reverse()
        data.correctionHorizontal()
        data.reverse()
        data.correctionHorizontal()
        data.reverse()
        # for i in field:
        #     data.target_field = i
        # if data.target_field == 'Temp':
        #     data.smooth()
        # elif data.target_field == 'Hum':
        #     data.smooth(gap=3 )
    data.normalization()
    data.outputToCsv('./arimadata/ARIMA_{}.csv'.format(file_name))
