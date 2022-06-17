import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime, timedelta
import os
from tqdm import tqdm
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
pd.set_option('mode.chained_assignment', None)


class NewChebyLSTM:
    target_field = 'Temp'
    direction = True

    def __init__(self, filename):
        df = pd.read_csv(filename, low_memory=False)
        # df = pd.read_csv(filename, low_memory=False)[:100000]
        self.data = df.copy()

    def normalization(self):
        self.data = self.data.round(2)

    def timeParser(self, time_format="dd/mm/YY HH:MM:SS"):
        LocalTime = self.data['LocalTime'].copy()
        l = []
        for j in tqdm(range(len(LocalTime))):
            if type(LocalTime[j]) != type('str'):
                l.append(j)
        self.data = self.data.drop(l, axis=0)
        if time_format == "dd/mm/YY HH:MM:SS":
            data = self.data['LocalTime'].copy()
            time = []
            for i in range(data.shape[0]):
                time_split = data.iloc[i].split(' ')
                date_split = time_split[0].split('/')
                year = '20'+date_split[2]
                month = date_split[0]
                day = date_split[1]
                today = year+'-'+month+'-'+day+' '+time_split[1]
                time.append(datetime.strptime(today, '%Y-%m-%d %H:%M:%S'))
            self.data['LocalTime'] = pd.Series(time)
            self.data = self.data.set_index('LocalTime')
        elif time_format == "YYYY/mm/dd HH:MM":
            data = self.data["LocalTime"].copy()
            time = []
            for i in range(self.data.shape[0]):
                time.append(datetime.strptime(self.data["LocalTime"][i].replace(
                    "/", "-")+":00", '%Y-%m-%d %H:%M:%S'))
            self.data['LocalTime'] = pd.Series(time)
            self.data = self.data.set_index('LocalTime')
        elif time_format == "YYYY-mm-dd HH:MM:SS":
            data = self.data["LocalTime"].copy()
            time = []
            for i in range(self.data.shape[0]):
                time.append(datetime.strptime(
                    self.data["LocalTime"][i], '%Y-%m-%d %H:%M:%S'))
            self.data['LocalTime'] = pd.Series(time)
            self.data = self.data.set_index('LocalTime')
        elif time_format == "YYYY/mm/dd HH:MM:SS":
            data = self.data["LocalTime"].copy()
            time = []
            for i in range(self.data.shape[0]):
                time.append(datetime.strptime(
                    self.data["LocalTime"][i].replace("/", "-"), '%Y-%m-%d %H:%M:%S'))
            self.data['LocalTime'] = pd.Series(time)
            self.data = self.data.set_index('LocalTime')

        elif time_format == "YYYY/m/dd HH:MM":
            data = self.data["LocalTime"].copy()
            time = []
            for i in range(data.shape[0]):
                time_split = data.iloc[i].split(' ')
                date_split = time_split[0].split('/')
                if len(date_split[1]) == 1:
                    new_day = "0{}".format(date_split[1])
                else:
                    new_day = date_split[1]
                date = date_split[0] + "-" + new_day + "-" + date_split[2] 
                hours = time_split[1] + ":00"
                today = date + " " + hours

                print(today)

                time.append(datetime.strptime(today, '%Y-%m-%d %H:%M:%S'))
            self.data['LocalTime'] = pd.Series(time)
            self.data = self.data.set_index('LocalTime')

    def newField(self):
        self.data[self.target_field + '_m'] = [np.nan] * self.data.shape[0]
        return self.target_field + '_m'

    def outputToCsv(self, filename, index=True):
        self.data.to_csv(filename, index=index)

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
                days += 1
        # self.scaler = MinMaxScaler(feature_range=(0, 1))
        # self.data = self.scaler.fit_transform(self.data)

    def set_condition(self):
        self.condition = np.isnan(self.data[self.target_field])
        self.condition = list(self.condition)

    def buildTrain(self, past_day=24, futureDay=1, day_data=288):
        self.x_train = []
        self.y_train = []
        if self.direction:
            for i in range(past_day+futureDay, self.data.shape[0]):
                if self.data[self.target_field][i-past_day-futureDay:i].count() < past_day+futureDay:
                    pass
                else:
                    x = np.array(
                        self.data[self.target_field].iloc[i-past_day-futureDay:i-futureDay])
                    x = x.reshape(x.shape[0], 1)
                    self.x_train.append(x)
                    self.y_train.append(
                        np.array(self.data.iloc[i-futureDay:i][self.target_field]))
        else:
            for i in range((past_day+futureDay)*day_data, self.data.shape[0]):
                # print(self.data.iloc[i-futureDay:i][self.target_field])
                if np.isnan(np.array(self.data[self.target_field].iloc[i-(past_day*day_data)-(futureDay*day_data):i-(futureDay*day_data):day_data])).sum() > 0:
                    pass
                elif np.isnan(self.data[i-futureDay:i][self.target_field].iloc[0]):
                    pass
                else:
                    x = np.array(self.data[self.target_field].iloc[i-(past_day*day_data)-(
                        futureDay*day_data):i-(futureDay*day_data):day_data])
                    x = x.reshape(x.shape[0], 1)
                    self.x_train.append(x)
                    self.y_train.append(
                        np.array(self.data.iloc[i-futureDay:i][self.target_field]))
        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)

    def shuffle(self, seed=10):
        np.random.seed(seed)
        randomList = np.arange(self.x_train.shape[0])
        np.random.shuffle(randomList)
        self.x_train = self.x_train[randomList]
        self.y_train = self.y_train[randomList]

    def buildModel(self):
        self.model = Sequential()
        self.model.add(
            LSTM(10, input_length=self.x_train.shape[1], input_dim=1))
        self.model.add(Dense(1))
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(loss="mse", optimizer=opt)
        self.model.summary()
        callback = EarlyStopping(
            monitor="loss", patience=10, verbose=1, mode="auto")
        self.x_train = self.x_train.astype(np.float32)
        self.y_train = self.y_train.astype(np.float32)
        # self.model.fit(self.x_train, self.y_train, epochs=3000,
        #                batch_size=128, callbacks=[callback])
        self.model.fit(self.x_train, self.y_train, epochs=10,
                        batch_size=1, callbacks=[callback])

    def modelSave(self, model_name='./models/lstm_model.h5'):
        self.model.save('./models/{}'.format(model_name))

    def modelLoad(self, model_name='./models/lstm_model.h5'):
        print(model_name)
        self.model = tf.keras.models.load_model('./models/{}'.format(model_name), compile=False)
        self.model.summary()

    def correction(self, past_day=24, correction_day=12, day_data=288):
        if self.direction:
            mask = list(np.isnan(self.data[self.target_field]))
            for i in tqdm(range(past_day+1, len(mask))):
                if mask[i]:
                    if self.condition[i - past_day:i].count(True) < correction_day:
                        if mask[i-past_day:i].count(True) == 0:
                            self.test = []
                            x = np.array(
                                self.data[self.target_field].iloc[i-past_day:i])
                            x = x.reshape(x.shape[0], 1)
                            self.test.append(x)
                            self.test = np.array(self.test)
                            y_hat = self.model.predict(self.test)
                            self.data[self.target_field][self.data.index[i]] = y_hat[0]
                            mask[i] = False
                            self.data[self.target_field +
                                      '_m'][self.data.index[i]] = 'V'
        else:
            mask = list(np.isnan(self.data[self.target_field]))
            for i in tqdm(range((past_day+1)*day_data, self.data.shape[0])):
                if mask[i]:
                    if self.condition[i - past_day*day_data:i:day_data].count(True) < correction_day:
                        if mask[i-past_day*day_data:i:day_data].count(True) == 0:
                            self.test = []
                            x = np.array(
                                self.data[self.target_field].iloc[i-past_day*day_data:i:day_data])
                            x = x.reshape(x.shape[0], 1)
                            self.test.append(x)
                            self.test = np.array(self.test)
                            y_hat = self.model.predict(self.test)
                            self.data[self.target_field][self.data.index[i]] = y_hat[0]
                            mask[i] = False
                            self.data[self.target_field +
                                      '_m'][self.data.index[i]] = 'H'

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

    def smooth(self, gap=0.3, fix=36):
        a_array = self.data[self.target_field].copy().tolist()
        b_array = a_array.copy()
        b_array.pop()
        b_array.insert(0, 0)
        diff_value = np.array(a_array) - np.array(b_array)
        diff_array = pd.Series((diff_value).tolist())
        positive_mask = diff_array >= gap
        negative_mask = diff_array <= -gap
        for i in tqdm(range(1, diff_array.shape[0])):
            if positive_mask[i] or negative_mask[i]:
                if diff_value[i] > 0:
                    x = diff_value[i] / fix
                    for j in range(fix):
                        k = fix - j
                        try:
                            if not np.isnan(self.data.iloc[i+j][self.target_field]) and self.data[self.target_field+"_m"][self.data.index[i + j]] == "H":
                                self.data[self.target_field][self.data.index[i + j]] -= x*k
                        except Exception as e:
                            pass
                        try:
                            if not np.isnan(self.data.iloc[i-j-1][self.target_field]) and self.data[self.target_field+"_m"][self.data.index[i-j-1]] == "H":
                                self.data[self.target_field][self.data.index[i-j-1]] += x*k
                        except Exception as e:
                            pass
                elif diff_value[i] < 0:
                    x = -diff_value[i] / fix
                    for j in range(fix):
                        k = fix - j
                        try:
                            if not np.isnan(self.data.iloc[i+j][self.target_field]) and self.data[self.target_field+"_m"][self.data.index[i + j]] == "H":
                                self.data[self.target_field][self.data.index[i + j]] += x*k
                        except Exception as e:
                            pass
                        try:
                            if not np.isnan(self.data.iloc[i-j-1][self.target_field]) and self.data[self.target_field+"_m"][self.data.index[i-j-1]] == "H":
                                self.data[self.target_field][self.data.index[i-j-1]] -= x*k
                        except Exception as e:
                            pass


if __name__ == "__main__":
    file_name = str(input("請輸入站別名稱："))
    print("目前運行站別：{}".format(file_name))
    if 1:
        print("load data start")
        data = NewChebyLSTM(file_name+'.csv')
        print("load data done")
        print("time parser start")
        if len(data.data['LocalTime'][0].split('/')[0]) == 2:
            data.timeParser()
        else:
            data.timeParser(time_format="YYYY/m/dd HH:MM")
        print("time parser done")
        print("anomaly detection start")
        data.normalization()
        data.anomalyDetection()
        data.target_field = 'Hum'
        data.anomalyDetection(value_maximun=100, value_minimun=30)
        data.target_field = 'Temp'
        print("anomaly detection done")
        data.outputToCsv('./chebydata/chebydata_{}.csv'.format(file_name))
        print(" chebydata 輸出完畢 ")
        data.buildTrain(past_day=24)
        data.shuffle()
        data.buildModel()
        data.modelSave(model_name="Temp_V_" + file_name + ".h5")
        data.target_field = 'Hum'
        data.buildTrain(past_day=24)
        data.shuffle()
        data.buildModel()
        data.modelSave(model_name="Hum_V_" + file_name + ".h5")
        # data.direction = False
        data.target_field = 'Temp'
        data.buildTrain(past_day=28)
        data.shuffle()
        data.buildModel()
        data.modelSave(model_name="Temp_H_" + file_name + ".h5")
        data.target_field = 'Hum'
        data.buildTrain(past_day=28)
        data.shuffle()
        data.buildModel()
        data.modelSave(model_name="Hum_H_" + file_name + ".h5")
        data.direction = True

        data.target_field = 'Temp'
        data.set_condition()
        data.modelLoad("Temp_V_{}.h5".format(file_name))
        data.newField()
        try:
            print(data.target_field+" vertical correction start")
            data.correction(past_day=24)
            data.reverse()
            print("reverse")
            data.correction(past_day=24)
            print(data.target_field+" vertical correction done")
        except Exception as e:
            print(e)
        data.reverse()

        data.direction = False
        data.modelLoad("Temp_H_{}.h5".format(file_name))
        try:
            print(data.target_field + "  horizontal correction start")
            data.correction(past_day=28, correction_day=14)
            data.reverse()
            print("reverse")
            data.correction(past_day=28, correction_day=14)
            print(data.target_field + " horizontal correction done")
        except Exception as e:
            print(e)
        data.reverse()
        data.direction = True
        data.target_field = 'Hum'
        data.set_condition()
        data.modelLoad("Hum_V_{}.h5".format(file_name))
        data.newField()
        try:
            print(data.target_field+" vertical correction start")
            data.correction(past_day=24)
            data.reverse()
            print("reverse")
            data.correction(past_day=24)
            print(data.target_field+" vertical correction done")
        except Exception as e:
            print(e)
        data.reverse()
        data.direction = False
        data.modelLoad("Hum_H_{}.h5".format(file_name))
        try:
            print(data.target_field+" horizontal correction start")
            data.correction(past_day=28, correction_day=14)
            data.reverse()
            print("reverse")
            data.correction(past_day=28, correction_day=14)
            print(data.target_field+" horizontal correction done")
        except Exception as e:
            print(e)
        data.reverse()
        data.normalization()
        data.outputToCsv('./LSTMdata/LSTM_{}.csv'.format(file_name))
        print(" LSTMdata 輸出完畢 ")
        data.target_field = 'Temp'
        data.smooth()
        data.target_field = "Hum"
        data.smooth(gap=3)
        data.outputToCsv('./smoothdata/smoothdata_{}.csv'.format(file_name))
        print(" smoothdata 輸出完畢 ")
