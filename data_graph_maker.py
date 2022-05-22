import numpy as np
import json
import matplotlib.pyplot as plt

X = []
Y1 = []
Y2 = []

l = 0

for i in range(1, 2, 1):
    with open('W' + str(i) + '_power7.json', 'r') as f:
        data = json.load(f)
        for d in data:
            X.append(l)
            l += 1
            Y1.append(abs(d[1][0]))
            Y2.append(abs(d[1][1]))
        
    plt.axvline(x=l-1, color='b', linewidth=0.3)
    
X = np.array(X)
Y1 = np.array(Y1)
Y2 = np.array(Y2)

plt.plot(X, Y1, 'r')
plt.plot(X, Y2, 'g')
plt.plot(X, (Y1 + Y2)/2, 'black')


plt.show()
