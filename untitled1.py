import numpy as np
import pylab as pl
import scipy.ndimage as nd
import skimage.io as io
from sklearn.cluster import KMeans
from sklearn import preprocessing
from matplotlib import cm


parma = io.imread("parma.jpg")
scaledparma = preprocessing.minmax_scale(parma.reshape(parma.shape[0]*parma.shape[1],3)
                                           .astype(float),axis=1)
 
clusters = 4
centers = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,1]])
kparma = KMeans(n_clusters=clusters, random_state=(123),init = centers).fit(scaledparma)


mycmap = cm.get_cmap('Set2', clusters)
pl.colorbar()
pl.axis('off')
qas