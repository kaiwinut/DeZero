import numpy as np
from dezero import utils
from dezero.core import Function, Variable, as_variable, as_array


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


class Reshape(Function):
	def __init__(self, shape):
		self.shape = shape

	def forward(self, x):
		self.x_shape = x.shape
		y = x.reshape(self.shape)
		return y

	def backward(self, gy):
		return reshape(gy, self.x_shape)


class Transpose(Function):
	def forward(self, x):
		y = np.transpose(x)
		return y

	def backward(self, gy):
		gx = transpose(gy)
		return gx


class Sum(Function):
	def __init__(self, axis, keepdims):
		self.axis = axis
		self.keepdims = keepdims

	def forward(self, x):
		self.x_shape = x.shape
		y = x.sum(axis = self.axis, keepdims = self.keepdims)
		return y

	def backward(self, gy):
		gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
		gx = broadcast_to(gy, self.x_shape)
		return gx


class BroadcastTo(Function):
	def __init__(self, shape):
		self.shape = shape

	def forward(self, x):
		self.x_shape = x.shape
		y = np.broadcast_to(x, self.shape)
		return y

	def backward(self, gy):
		# For example, y = x + x, when x is used twice, 
		# the gradient is the sum of the two variables used
		gx = sum_to(gy, self.x_shape)
		return gx


class SumTo(Function):
	def __init__(self, shape):
		self.shape = shape

	def forward(self, x):
		self.x_shape = x.shape
		y = utils.sum_to(x, self.shape)
		return y

	def backward(self, gy):
		gx = broadcast_to(gy, self.x_shape)
		return gx


def sin(x):
	return Sin()(x)


def cos(x):
	return Cos()(x)


def tanh(x):
	return Tanh()(x)


def reshape(x, shape):
	if x.shape == shape:
		return as_variable(x)
	return Reshape(shape)(x)


def transpose(x):
	return Transpose()(x)


def sum(x, axis = None, keepdims = False):
	return Sum(axis, keepdims)(x)


def broadcast_to(x, shape):
	if x.shape == shape:
		return as_variable(x)
	return BroadcastTo(shape)(x)


def sum_to(x, shape):
	if x.shape == shape:
		return as_variable(x)
	return SumTo(shape)(x)