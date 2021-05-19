import numpy as np
import pylab as pl 



X = np.array([ [0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1] ])

y = np.array([[0,1,1,0]]).T
print("predict:\n", y)
print("based on:\n", X)


def sigmoid(x):
  return 1. / (1 + np.exp(-x))

def dsigmoid(x):
  return np.exp(-x)/((1+np.exp(-x)**2))

def loss(predict):
    return y - predict

training = 6000
np.random.seed(123)

# randomly initialize our weights with mean 0
syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

print (syn0.shape, "\n", syn1.shape)

loss_hidden = []
loss_output = []

# 2 layers
for iterate in range(training):
  inputLayer = X  
  #dot product
  la1 = np.dot(inputLayer, syn0)
  #activate
  layer1out = sigmoid(la1)

  la2 = np.dot(layer1out, syn1)
  #activate
  outputLayer = sigmoid(la2)
  
  #calculate loss on the output layer
  outputLayer_error = loss(outputLayer)
  loss_output.append(outputLayer_error.sum())
  outputLayer_delta = outputLayer_error * dsigmoid(outputLayer)

  l1_error = outputLayer_delta.dot(syn1.T)
  loss_hidden.append(l1_error.sum())
  l1_delta = l1_error * dsigmoid(layer1out)
 
  # back propagation step
  # multiply how much we missed by the
  # slope of the sigmoid at the values in l1

  syn1 += outputLayer.T.dot(outputLayer_delta)
  syn0 += inputLayer.T.dot(l1_delta)
  
print ("Final Prediction:\n", outputLayer)

print ("target: \n", y)

