from imputpredicter.newchebyLSTM import NewChebyLSTM 
import os
import numpy as np
import pandas as pd
from random import randint

def test_data(l, f):
    nan_list = []
    data_num = []
    for i in range(len(l)):
        if np.isnan(l[f][l.index[i]]):
            nan_list.append(i)
    for i in range(1, len(nan_list)):
        data_num.append(nan_list[i]-nan_list[i-1])
    max_index = data_num.index(max(data_num))
    return l[nan_list[max_index]+1:nan_list[max_index+1]].copy().reset_index().set_index('LocalTime')

def get_mse(l1, l2):
    s = 0
    for i in range(len(l1)):
        s += (l1[i]-l2[i]) ** 2
    return (s / len(l1))

def del_data(l, f):
    for i in range(len(l)):
        a = randint(0, 30)
        print(a)
        if a == 0:
            l[f][l.index[i]] = np.nan
    return l.copy()


if __name__ == "__main__":
    files = os.listdir("./new10dataNOna/")
    
    mse = {"local":[], "temp":[], "hum":[]}
    for i in files:
        try:
            print(i)
            print("load data start")
            # data = NewChebyLSTM("../enddata/" + i)
            data = NewChebyLSTM("./new10dataNOna/" + i)

            mse["local"].append(i)

            print("load data done")
            print("time parser start")
            if len(data.data['LocalTime'][0].split('/')[0]) == 2:
                data.timeParser()
            else:
                data.timeParser(time_format="YYYY/mm/dd HH:MM")
            print("time parser done")
            print("anomaly detection start")
            data.normalization()
            data.anomalyDetection()
            data.target_field = 'Hum'
            data.anomalyDetection(value_maximun = 100, value_minimun = 30)
            data.target_field = 'Temp'
            print("anomaly detection done")
            
            source_data = data.data.copy()
            data.data = test_data(data.data.copy(), "Temp")
            true_data = data.data.copy()
            data.data = del_data(data.data.copy(), "Temp")

            data.direction = True
            data.target_field = 'Temp'
            data.set_condition()
            data.modelLoad("./model/Temp_V_"+"01_原臺南州廳綜合氣象站.csv"+".h5")
            data.newField()
            try:
                print(data.target_field+" vertical correction start")
                data.correction(past_day = 24)
                data.reverse()
                print("reverse")
                data.correction(past_day = 24)
                print(data.target_field+" vertical correction done")
            except Exception as e:
                print(e)
                data.reverse()

            print(true_data["Temp"])
            print(data.data["Temp"])
            # input()
            mse["temp"].append(get_mse(true_data["Temp"].tolist(), data.data["Temp"].tolist()))
            print(get_mse(true_data["Temp"].tolist(), data.data["Temp"].tolist()))

            # data.direction = False
            # data.modelLoad("./model/Temp_H_"+"01_原臺南州廳綜合氣象站.csv"+".h5")
            # try:
            #     print(data.target_field+" horizontal correction start")
            #     data.correction(past_day = 28, correction_day=14, day_data=48)
            #     data.reverse()
            #     print("reverse")
            #     data.correction(past_day = 28, correction_day=14, day_data=48)
            #     print(data.target_field+" horizontal correction done")
            # except Exception as e:
            #     print(e)
            #     data.reverse()

            data.data = source_data.copy()
            data.data = test_data(data.data.copy(), "Hum")
            true_data = data.data.copy()
            data.data = del_data(data.data.copy(), "Hum")

            data.direction = True
            data.target_field = 'Hum'
            data.set_condition()
            data.modelLoad("./model/Hum_V_"+"01_原臺南州廳綜合氣象站.csv"+".h5")
            data.newField()
            try:
                print(data.target_field+" vertical correction start")
                data.correction(past_day = 24)
                data.reverse()
                print("reverse")
                data.correction(past_day = 24)
                print(data.target_field+" vertical correction done")
            except Exception as e:
                print(e)
                data.reverse()

            mse["hum"].append(get_mse(true_data["Hum"].tolist(), data.data["Hum"].tolist()))

            mseD = pd.DataFrame(mse)
            mseD.to_csv("./LSTM_MSE.csv")

            # data.direction = False
            # data.modelLoad("./model/Hum_H_"+"01_原臺南州廳綜合氣象站.csv"+".h5")
            # try:
            #     print(data.target_field+" horizontal correction start")
            #     data.correction(past_day = 28, correction_day=14, day_data=48)
            #     data.reverse()
            #     print("reverse")
            #     data.correction(past_day = 28, correction_day=14, day_data=48)
            #     print(data.target_field+" horizontal correction done")
            # except Exception as e:
            #     print(e)
            #     data.reverse()

            # data.normalization()
            # data.outputToCsv("../LSTM/1000LSTM_"+str(i))
            # data.outputToCsv("../30min/donedata/"+i+".csv")
            # print("output done")

            # data = NewChebyLSTM("../LSTM/" + i)
            # data.timeParser(time_format="YYYY-mm-dd HH:MM:SS")
            # data.target_field = 'Temp'
            # data.smooth()
            # data.target_field = "Hum"
            # data.smooth(gap = 3)
            # data.outputToCsv("../30min/smoothdata/"+str(i))

        except Exception as e:
            print(e)