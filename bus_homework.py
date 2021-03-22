import numpy as np

tt = np.load("busTimeTable.npy")
busses = np.load("busBusses.npy")
hours = range(1,11)

avgTrip = tt.mean(axis=0)

longestHour = np.array(hours)[avgTrip == avgTrip.max()]

print("The busiest hour is " + str(longestHour) + " P.M")

slowestBus = busses[tt.mean(axis=1) == tt.mean(axis=1).max()]
print("The slowest bus is bus n° " + str(slowestBus))