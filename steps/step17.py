import weakref
import numpy as np

"""
a = np.array([1, 2, 3])
b = weakref.ref(a)
print(b)
print(b())
# <weakref at 0x1051408b0; to 'numpy.ndarray' at 0x105142630>
# [1 2 3]

a = None
print(b)
# <weakref at 0x1051408b0; dead>
"""

class Variable:
	def __init__(self, data):
		if data is not None:
			if not isinstance(data, np.ndarray):
				raise TypeError('{} is not supported'.format(type(data)))

		self.data = data
		self.grad = None
		self.creator = None

		# Manages the order backpropagation is called
		self.generation = 0

	def set_creator(self, func):
		self.creator = func
		self.generation = func.generation + 1

	def backward(self):
		# The gradient of the output variable will always be 1.0
		if self.grad is None:
			self.grad = np.ones_like(self.data)

		funcs = []
		seen_set = set()

		def add_func(f):
			if f not in seen_set:
				funcs.append(f)
				seen_set.add(f)
				# This can be accelerated by using heapq in python
				funcs.sort(key = lambda x: x.generation)

		add_func(self.creator)

		# Perform backpropagation with iterations
		while funcs:
			f = funcs.pop()
			gys = [output().grad for output in f.outputs]
			gxs = f.backward(*gys)
			if not isinstance(gxs, tuple):
				gxs = (gxs,)

			for x, gx in zip(f.inputs, gxs):
				if x.grad is None:
					x.grad = gx
				else:
					x.grad = x.grad + gx


				if x.creator is not None:
					add_func(x.creator)

	def cleargrad(self):
		self.grad = None


class Function:
	def __call__(self, *inputs):
		xs = [x.data for x in inputs]
		ys = self.forward(*xs)
		if not isinstance(ys, tuple):
			ys = (ys,)
		outputs = [Variable(as_array(y)) for y in ys]

		self.generation = max([x.generation for x in inputs])
		for output in outputs:
			output.set_creator(self)

		self.inputs = inputs
		self.outputs = [weakref.ref(output) for output in outputs]
		return outputs if len(outputs) > 1 else outputs[0]

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
		x = self.inputs[0].data
		gx = 2 * x * gy
		return gx


class Exp(Function):
	def forward(self, x):
		return np.exp(x)

	def backward(self, gy):
		x = self.input.data
		gx = np.exp(x) * gy
		return gx


class Add(Function):
	def forward(self, x0, x1):
		y = x0 + x1
		return (y,)

	def backward(self, gy):
		return gy, gy


def square(x):
	return Square()(x)


def exp(x):
	return Exp()(x)


def add(x0, x1):
	return Add()(x0, x1)


def as_array(x):
	if np.isscalar(x):
		return np.array(x)
	return x


if __name__ == '__main__':
	# Memory will be freed after each loop
	for i in range(10):
		x = Variable(np.random.randn(10000))
		y = square(square(square(x)))