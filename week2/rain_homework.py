import pandas as pd
import numpy as np
import scipy.stats as sp

df = pd.read_csv("https://raw.githubusercontent.com/fedhere/DSPS/master/labs/1865331.csv")

listnumv = ["PRCP", "SNOW", "SNWD", "WESD","WES"]


samp = df.PRCP[pd.to_datetime(df.DATE) > pd.to_datetime("2018-12-31")]
pop = df.PRCP[~(pd.to_datetime(df.DATE) > pd.to_datetime("2018-12-31"))]

def Z(pop,samp):
    return (pop.mean() - samp.mean())/(np.std(pop) / np.sqrt(len(samp)))

Zscore = Z(pop,samp)
print("The calculated Z-score is " + str(Zscore))

print("The corresponding probability is " + str(sp.norm(0,1).pdf(Zscore)))