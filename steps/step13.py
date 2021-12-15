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
			gys = [output.grad for output in f.outputs]
			gxs = f.backward(*gys)
			if not isinstance(gxs, tuple):
				gxs = (gxs,)

			for x, gx in zip(f.inputs, gxs):
				x.grad = gx

				if x.creator is not None:
					funcs.append(x.creator)


class Function:
	def __call__(self, *inputs):
		xs = [x.data for x in inputs]
		ys = self.forward(*xs)
		if not isinstance(ys, tuple):
			ys = (ys,)
		outputs = [Variable(as_array(y)) for y in ys]

		for output in outputs:
			output.set_creator(self)

		self.inputs = inputs
		self.outputs = outputs
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
	# x = Variable(np.array(2.0)) 
	# y = Variable(np.array(3.0))
	# z = add(square(x), square(y))
	# z.backward()

	# print(z.data)
	# print(x.grad)
	# print(y.grad)

	x = Variable(np.array(3.0))
	y = add(x, x)
	print('y', y.data)

	y.backward()
	# x.grad should be 2.0 in this case. However the output is 1.0.
	print('x.grad', x.grad)