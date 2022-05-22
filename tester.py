import numpy as np
from keras.models import Sequential
from keras.layers import Dense #필요한 라이브러리 참조
from keras.models import load_model
import json


model2 = load_model('7_trex_ann_model.h5')
#model2 = load_model('2_trex_ann_model.h5')

w = np.array(model2.get_weights())

dense_1 = w[0]
bias_1 = w[1]
dense_2 = w[2]
bias_2 = w[3]
#dense_3 = w[4]
#bias_3 = w[5]
#dense_4 = w[6]
#bias_4 = w[7]


size = 10

W1 = np.zeros((size, ))
data = []

for i in range(0, size):
    WW1 = W1
    WW1[i] = 1
    #WW1 = np.matmul(WW1, dense_2) + bias_2
    #WW1 = np.matmul(WW1, dense_3) + bias_3

    Y = np.matmul(WW1, dense_2) + bias_2

    data.append([i, [float(Y[0]), float(Y[1])]]) #

with open('W1_power7.json', 'w') as f:
    json.dump(data, f)