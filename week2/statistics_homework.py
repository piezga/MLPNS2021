from __future__ import print_function, division
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


maxSize = 300

sizes = np.arange(1,maxSize)

means = np.array([])
for i in range(1,maxSize):
    numbers = stats.poisson.rvs(mu = 8, size = i*10)
    mean = numbers.mean()
    means = np.append(means,mean)
    


plt.scatter(sizes,means)
plt.title('I used the poisson distribution')
plt.xlabel('Sample size/10')
plt.ylabel('Calculated mean')
plt.axhline(8, color ='r')
plt.show()