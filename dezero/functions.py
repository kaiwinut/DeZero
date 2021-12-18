import numpy as np
from dezero.core import Function


class Sin(Function):
	def forward(self, x):
		y = np.sin(x)
		return y

	def backward(self, gy):
		x, = self.inputs
		gx = gy * cos(x)
		return gx


class Cos(Function):
	def forward(self, x):
		y = np.cos(x)
		return y

	def backward(self, gy):
		x, = self.inputs
		gx = gy * -sin(x)
		return gx


class Tanh(Function):
	def forward(self, x):
		y = np.tanh(x)
		return y

	def backward(self, gy):
		y = self.outputs[0]()
		gx = gy * (1 - y * y)
		return gx


def sin(x):
	return Sin()(x)


def cos(x):
	return Cos()(x)


def tanh(x):
	return Tanh()(x)