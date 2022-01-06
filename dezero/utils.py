import os
import subprocess
import numpy as np

def _dot_var(v, verbose = False):
	dot_var = '{} [label="{}", color=orange, style=filled]\n'

	name = '' if v.name is None else v.name
	if verbose and v.data is not None:
		if v.name is not None:
			name += ': '
		name += str(v.shape) + ' ' + str(v.dtype)
	return dot_var.format(id(v), name)


def _dot_func(f):
	dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
	txt = dot_func.format(id(f), f.__class__.__name__)

	dot_edge = '{} -> {}\n'

	for x in f.inputs:
		txt += dot_edge.format(id(x), id(f))

	for y in f.outputs:
		txt += dot_edge.format(id(f), id(y()))

	return txt


def get_dot_graph(output, verbose = True):
	txt = ''
	funcs = []
	seen_set = set()

	def add_func(f):
		if f not in seen_set:
			funcs.append(f)
			seen_set.add(f)

	add_func(output.creator)
	txt += _dot_var(output, verbose)

	while funcs:
		func = funcs.pop()
		txt += _dot_func(func)
		for x in func.inputs:
			txt += _dot_var(x, verbose)

			if x.creator is not None:
				add_func(x.creator)

	return 'digraph g {\n' + txt + '}'


def plot_dot_graph(output, verbose = True, to_file = 'graph.png'):
	dot_graph = get_dot_graph(output, verbose)

	tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
	if not os.path.exists(tmp_dir):
		# ~/.dezero
		os.mkdir(tmp_dir)
	graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

	with open(graph_path, 'w') as f:
		f.write(dot_graph)

	# e.g. png, pdf, etc.
	extension = os.path.splitext(to_file)[1][1:]
	cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
	subprocess.run(cmd, shell=True)


####################################
#								   #
#    Implementation not in book    #
#								   #
####################################
def sum_to(x, shape):
	ndim = len(shape)
	lead = x.ndim - ndim
	lead_axis = tuple(range(lead))

	axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
	y = x.sum(lead_axis + axis, keepdims = True)
	if lead > 0:
		y = y.squeeze(lead_axis)
	return y


def reshape_sum_backward(gy, x_shape, axis, keepdims):
	ndim = len(x_shape)
	tupled_axis = axis
	if axis is None:
		tupled_axis = None
	elif not isinstance(axis, tuple):
		tupled_axis = (axis,)

	if not (ndim == 0 or tupled_axis is None or keepdims):
		actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
		shape = list(gy.shape)
		for a in sorted(actual_axis):
			shape.insert(a, 1)

	else:
		shape = gy.shape

	gy = gy.reshape(shape)
	return gy


def max_backward_shape(x, axis):
	if axis is None:
		axis = range(x.ndim)
	elif isinstance(axis, int):
		axis = (axis,)
	else:
		axis = axis

	shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]
	return shape


def logsumexp(x, axis=1):
	m = x.max(axis=axis, keepdims=True)
	y = x - m
	np.exp(y, out=y)
	s = y.sum(axis=axis, keepdims=True)
	np.log(s, out=s)
	m += s
	return m


def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError