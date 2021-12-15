import weakref
import contextlib
import numpy as np


class Config:
	enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
	# Process that will be executed when the "with" clause is called
	old_value = getattr(Config, name)
	setattr(Config, name, value)
	# the border
	try:
		yield
	# Process that will be executed after the "with" clause
	finally:
		setattr(Config, name, old_value)


def no_grad():
	return using_config('enable_backprop', False)


class Variable:
	def __init__(self, data, name = None):
		if data is not None:
			if not isinstance(data, np.ndarray):
				raise TypeError('{} is not supported'.format(type(data)))

		self.data = data
		self.name = name
		self.grad = None
		self.creator = None

		# Manages the order backpropagation is called
		self.generation = 0

	def set_creator(self, func):
		self.creator = func
		self.generation = func.generation + 1

	def backward(self, retain_grad = False):
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

			if not retain_grad:
				for y in f.outputs:
					y().grad = None

	def cleargrad(self):
		self.grad = None	

	# Enables user to access the shape property with x.shape instead of x.shape()
	@property	
	def shape(self):
		return self.data.shape

	@property	
	def ndim(self):
		return self.data.ndim

	@property	
	def size(self):
		return self.data.size

	@property	
	def dtype(self):
		return self.data.dtype

	def __len__(self):
		return len(self.data)

	def __repr__(self):
		if self.data is None:
			return 'variable(None)'
		p = str(self.data).replace('\n', '\n' + ' ' * 9)
		return 'variable(' + p + ')'

class Function:
	def __call__(self, *inputs):
		xs = [x.data for x in inputs]
		ys = self.forward(*xs)
		if not isinstance(ys, tuple):
			ys = (ys,)
		outputs = [Variable(as_array(y)) for y in ys]

		if Config.enable_backprop:
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
	x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
	print(x)