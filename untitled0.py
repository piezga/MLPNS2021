import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn import cluster
import scipy as sp
from scipy import spatial

#index i <-> gene i+1

genes = pd.read_csv("https://raw.githubusercontent.com/fedhere/DSPS/master/HW10/kidpackgenes.csv",
                    index_col=(0))
scaledgenes = skl.preprocessing.scale(genes)

#projection
