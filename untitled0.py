import scipy.stats as sps
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

eqdata = pd.read_csv("https://raw.githubusercontent.com/fedhere/MLPNS2021/main/NHRT/earthquakes.csv", sep=" ")

#rename columns

eqdata.rename({"#YYY/MM/DD" : "date",
               "Unnamed: 5" : "mag",
               "HH:mm:SS.ss": "time"},axis=1)[["date","time","mag"]]

#data['HH:mm:SS.ss'] = data['HH:mm:SS.ss'].str.replace('60.00', '59.99')

for i in range(len(eqdata["time"])):
    if eqdata.iloc[i].time.endswith("60.00"):
        print(eqdata.iloc[i].time)
        print(eqdata.iloc[i].time[:6] + '59.99')
        eqdata["time"][i] = eqdata.iloc[i].time[:6] + "59.99"
        

pd.to_datetime(eqdata['time'])