if '__file__' in globals():
	import os, sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable, as_variable
import dezero.functions as F
from dezero.models import MLP


def softmax1d(x):
	x = as_variable(x)
	y = F.exp(x)
	sum_y = F.sum(y)
	return y / sum_y


model = MLP((10, 3))
x = Variable(np.array([[0.2, -0.4]]))
y = model(x)
p = F.softmax(y)
# p = softmax1d(y)
p.backward()

print(x.grad)
print(p)