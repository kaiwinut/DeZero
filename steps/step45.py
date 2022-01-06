if '__file__' in globals():
	import os, sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dezero import Variable, Model
import dezero.functions as F
import dezero.layers as L
from dezero.models import MLP

# Dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
max_iter = 10000
hidden_size = 10

# Model definition
class TwoLayerNet(Model):
	def __init__(self, hidden_size, out_size):
		super().__init__()
		self.l1 = L.Linear(hidden_size)
		self.l2 = L.Linear(out_size)

	def forward(self, x):
		y = F.sigmoid(self.l1(x))
		y = self.l2(y)
		return y

# model = TwoLayerNet(hidden_size, 1)
model = MLP((10, 1))

# Training process
n_frame = 100
fig = plt.figure(figsize=(7, 5))
plt.style.use('ggplot')
x_lin = np.arange(0.0, 1.0, 0.01)
y_tracker = []
loss_tracker = []

def update(i):
	# determine the results the i-th frame uses
	idx = i * max_iter // n_frame

	# plot
	plt.cla()
	plt.scatter(x, y, c="skyblue", label="data")
	plt.plot(x_lin, y_tracker[idx], c="red", label="prediction")
	plt.xlabel('x')
	plt.ylabel('y')
	plt.suptitle('Two Layer Network', fontsize = 14)
	plt.title('iter: ' + str(idx) + 
			  ', loss: ' + str(np.round(loss_tracker[idx], 4)) + 
			  ', lr: ' + str(lr) +
			  ', N: ' + str(len(x)), loc='left', fontsize = 8)
	plt.legend()
	plt.grid(color="white", linestyle="solid")


for i in range(max_iter):
	y_pred = model(x)
	y_lin = model(x_lin.reshape(x.shape))
	loss = F.mean_squared_error(y, y_pred)

	model.cleargrads()
	loss.backward()

	for p in model.params():
		p.data -= lr * p.grad.data

	# if i % (max_iter // n_frame) == 0:
		# print(loss)

	y_tracker.append(y_lin.data)
	loss_tracker.append(loss.data)


ani = animation.FuncAnimation(fig, update, frames = n_frame, interval = 50)
plt.show()