import numpy as np


class Variable:
	def __init__(self, data):
		# Only allow np.ndarray inputs
		if data is not None:
			if not isinstance(data, np.ndarray):
				raise TypeError('{} is not supported'.format(type(data)))

		self.data = data
		self.grad = None
		self.creator = None

	def set_creator(self, func):
		self.creator = func

	def backward(self):
		# The gradient of the output variable will always be 1.0
		if self.grad is None:
			self.grad = np.ones_like(self.data)

		funcs = [self.creator]

		# Perform backpropagation with iterations
		while funcs:
			f = funcs.pop()
			x, y = f.input, f.output
			x.grad = f.backward(y.grad)

			if x.creator is not None:
				funcs.append(x.creator)

class Function:
	def __call__(self, input):
		x = input.data
		# y might become a scalar value even if the input 
		# of the forward function x is an ndarray
		y = self.forward(x)
		output = Variable(as_array(y))
		output.set_creator(self)
		self.input = input
		self.output = output
		return output

	# If the forward function of a child of the Function class is not implemented,
	# NotImplementedError will be raised
	def forward(self, x):
		raise NotImplementedError()

	# If the backward function of a child of the Function class is not implemented,
	# NotImplementedError will be raised
	def backward(self, gy):
		raise NotImplementedError()


class Square(Function):
	def forward(self, x):
		return x ** 2

	def backward(self, gy):
		x = self.input.data
		gx = 2 * x * gy
		return gx


class Exp(Function):
	def forward(self, x):
		return np.exp(x)

	def backward(self, gy):
		x = self.input.data
		gx = np.exp(x) * gy
		return gx


def square(x):
	return Square()(x)


def exp(x):
	return Exp()(x)


def as_array(x):
	if np.isscalar(x):
		return np.array(x)
	return x

if __name__ == '__main__':
	x = Variable(np.array(0.5))
	y = square(exp(square(x)))

	y.backward()
	print(x.grad)