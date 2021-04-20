import pandas as pd 
import pylab as pl
import numpy as np
from sklearn import linear_model
from scipy.optimize import minimize
from scipy.optimize import least_squares as lsq

grbAG = pd.read_csv("https://raw.githubusercontent.com/fedhere/MLPNS2021/main/HW3_LinearRegression/grb050525A.csv")

grbAG["logtime"] = np.log10(grbAG["time"])

#visualize the data
ax = pl.figure(figsize=(10,10)).add_subplot(111)

for f in grbAG["filter"].unique():
    pl.errorbar(grbAG.loc[grbAG["filter"] == f, "logtime"], 
                grbAG.loc[grbAG["filter"] == f, "mag"], 
                yerr=grbAG.loc[grbAG["filter"] == f, "magerr"], fmt='.', ms=0,
                fcolor=None, label=None)

# replot to add a better marker (optional)
for f in grbAG["filter"].unique():
    pl.scatter(grbAG.loc[grbAG["filter"] == f, "logtime"], 
               grbAG.loc[grbAG["filter"] == f, "mag"], 
               alpha=1, s=75, linewidth=2,
               label=f,
               edgecolor='#cccccc')
    
# plot the upperlimits as arrows
nuplim = grbAG.mag.isna().sum()
for i in grbAG[grbAG.upperlimit == 1].index:
    pl.arrow(grbAG.loc[i].logtime, 
             grbAG.loc[i].magerr, 0, 2, 
            head_width=0.05, head_length=0.1, ec='k')
    
pl.ylim(24,11)
pl.legend()
pl.xlabel("Log(time)", fontsize = 20)
pl.ylabel("Magnitude", fontsize = 20)

cleanData = grbAG[grbAG.upperlimit == 0]

x1 = np.array([cleanData.logtime]).reshape(-1,1)
y1 = np.array([cleanData.mag]).reshape(-1,1)

regr = linear_model.LinearRegression()
regr.fit(x1,y1)

print ("best fit parameters from the sklearn LinearRegression():" 
       + "slope = " + str(regr.coef_) + " intercept = " + str(regr.intercept_))

yplot = regr.coef_*x1+ regr.intercept_
pl.plot(x1, yplot, color = 'black', linewidth = 3, label = 'sklearn')

#fit made with function minimization

def line(x, m, q):
    return m*x + q

def residual(p,x,y):
    return y - line(x,*p)

guess = (3,3)

x2 = np.array([cleanData.logtime])
y2 = np.array([cleanData.mag])

xfit = x2[0,:]
yfit = y2[0,:]

m, q = np.polyfit(xfit,yfit,1)

print("Fit parameters obtained from polyfit: m,q =" 
      + str(m) +" " + str(q))

pl.plot(x1,line(x1,m,q), color = 'red', linewidth = 1, label = 'polyfit')
pl.legend()

msq, qsq = lsq(residual,guess,args = (xfit,yfit))



