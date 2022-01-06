if '__file__' in globals():
	import os, sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import dezero.functions as F
import dezero.layers as L

# Dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

l1 = L.Linear(10)
l2 = L.Linear(1)

def predict(x):
	y = l1(x)
	y = F.sigmoid(y)
	y = l2(y)
	return y

lr = 0.2
iters = 10000

fig = plt.figure()
ims = []

for i in range(iters):
	y_pred = predict(x)
	loss = F.mean_squared_error(y, y_pred)

	l1.cleargrads()
	l2.cleargrads()
	loss.backward()

	for l in [l1, l2]:
		for p in l.params():
			p.data -= lr * p.grad.data

	if i % 100 == 0:
		# print(loss)
		im = plt.scatter(x, predict(x).data, marker = "x", c = "r")
		plt.scatter(x, y)
		ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval = 10, repeat_delay = 1000)
plt.show()