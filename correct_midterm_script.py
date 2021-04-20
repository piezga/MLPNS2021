import numpy as np
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import k_means

initialData = pd.read_excel("/home/pietro/Documents/Uni/MLPNS2021/country_pop.xls",
                            header = 3, usecols = "A,E:BL", index_col=(0))

#counting lost observations
countryCounter = 0
dataCounter = 0
for i in initialData.index:
    dataCounter = dataCounter + initialData.loc[i].isna().sum()
    if initialData.loc[i].isna().sum() != 0:
        countryCounter = countryCounter + 1
        
removedPerc = dataCounter/(len(initialData)*len(initialData.T))*100
        
print("The number of countries that will be removed is " + str(countryCounter))
print("The number of empty data entries is " + str(dataCounter))
print("The percentage of removed data is around %0.2f"%(removedPerc) + "%") 

cleanData = initialData.dropna()

popvals = np.array([cleanData.loc[i].values],float).reshape(60,1)
logpopvals = np.log(np.array([cleanData.loc[i].values],float).reshape(60,1))

for i in cleanData.index:
    popplot = plt.plot(np.array([cleanData.loc[i].values],float).reshape(60,1))
    plt.title("Population of world by country")
    plt.xlabel("Years since 1960")
plt.show()

for i in cleanData.index:
    popplot = plt.plot(np.log(np.array([cleanData.loc[i].values],float)).reshape(60,1))
    plt.title("Population of world by country (log)")
    plt.xlabel("Years since 1960")
plt.show()

rows = cleanData.shape[0]
columns = cleanData.shape[1]
procdata = np.array([[0.0]*columns]*rows)
ndivprocdata = np.array([[0.0]*columns]*rows)

for i in range(len(cleanData.index)):
    ndivprocdata[i,:] = (np.array([cleanData.iloc[i].values]) - 
                     np.array([cleanData.iloc[i].values]).mean())
    procdata[i,:] = np.divide(ndivprocdata[i,:],np.std(ndivprocdata[i,:]))
    
def parab(x,a,b,c):
    return a*x**2 + b*x + c
    
x = np.arange(0,60)
features = np.array([[0.0]*3]*rows)
for i in range(features.shape[0]):
    fit = np.polyfit(x,procdata[i,:],2)
    features[i,:] = fit

procfeatures = np.array([[0.0]*3]*rows)

for i in range(rows):
    procfeatures[i,:] = np.divide(features[i,:] - features[i,:].mean(), np.std(features[i,:]))
    

c = k_means(procfeatures,4,random_state=(123))

c0 = cleanData[c[1] == 0]
c1 = cleanData[c[1] == 1]
c2 = cleanData[c[1] == 2]
c3 = cleanData[c[1] == 3]

print("Countries belonging to cluster 0 are: " +
      str(c0.index))
print("Countries belonging to cluster 1 are: " +
      str(c1.index))
print("Countries belonging to cluster 2 are: " +
      str(c2.index))
print("Countries belonging to cluster 3 are: " +
      str(c3.index))


for i in c0.index:
    popplot = plt.plot(np.array([c0.loc[i].values],float).reshape(60,1))
    plt.title("Cluster 0 population")
    plt.xlabel("Years since 1960")
plt.show()

for i in c1.index:
    popplot = plt.plot(np.array([c1.loc[i].values],float).reshape(60,1))
    plt.title("Cluster 1 population")
    plt.xlabel("Years since 1960")
plt.show()

for i in c2.index:
    popplot = plt.plot(np.array([c2.loc[i].values],float).reshape(60,1))
    plt.title("Cluster 2 population")
    plt.xlabel("Years since 1960")
plt.show()

for i in c3.index:
    popplot = plt.plot(np.array([c3.loc[i].values],float).reshape(60,1))
    plt.title("Cluster 0 population")
    plt.xlabel("Years since 1960")
plt.show()

