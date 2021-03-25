import pandas as pd
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/fedhere/DSPS/master/labs/1865331.csv")

listnumv = ["PRCP", "SNOW", "SNWD", "WESD","WES"]


samp = df.DATE[pd.to_datetime(df.DATE) > pd.to_datetime("2018-12-31")]
pop = df.DATE[~(pd.to_datetime(df.DATE) > pd.to_datetime("2018-12-31"))]

def Z(pop,samp):
    return (pop.mean() - samp.mean())/(np.stdev(pop)) / np.sqrt(len(samp))

print(Z(pop.PRCP, samp.PRCP))