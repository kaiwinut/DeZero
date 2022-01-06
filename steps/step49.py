if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
import dezero


# Graph
markers = ['o', 'x', '^']
x0_lin = np.arange(-1.1, 1.1, 0.01)
x1_lin = np.arange(-1.1, 1.1, 0.01)
x0_grid, x1_grid = np.meshgrid(x0_lin, x1_lin)
x_points = np.c_[x0_grid.ravel(), x1_grid.ravel()]
y_grids = []
loss_tracker = []
fig = plt.figure()

def update(i):
    plt.cla()
    plt.contourf(x0_grid, x1_grid, y_grids[i])
    for c in range(3):
        plt.scatter(x[t == c, 0], x[t == c, 1], marker=markers[c], s=50, label='class '+str(c))
    plt.xlabel('$x_0$', fontsize=8)
    plt.ylabel('$x_1$', fontsize=8)
    plt.suptitle('Spiral', fontsize=14)
    plt.title('iter:' + str(i + 1) + ', loss=' + str(np.round(loss_tracker[i], 4)), loc='left')
    plt.legend()


# Training
max_epoch = 150
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = dezero.datasets.Spiral()
# train_set = dezero.datasets.Spiral(transform = lambda x: x/2)
x = np.array([d[0] for d in [train_set[i] for i in range(len(train_set))]])
t = np.array([d[1] for d in [train_set[i] for i in range(len(train_set))]])

model = MLP((hidden_size, 3))
optimizer = optimizers.MomentumSGD(lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size: (i+1) * batch_size]
        batch = [train_set[i] for i in batch_index]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])

        y = model(batch_x)
        y_grids.append(np.argmax(model(x_points).data, axis=1).reshape(x0_grid.shape))
        # loss = F.softmax_cross_entropy_simple(y, batch_t)
        loss = F.softmax_cross_entropy(y, batch_t)        
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))
    loss_tracker.append(avg_loss)

ani = animation.FuncAnimation(fig, update, frames=max_epoch, interval=100)
plt.show()