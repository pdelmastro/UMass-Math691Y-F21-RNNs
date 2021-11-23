"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch
Documentation: https://deepxde.readthedocs.io/en/latest/demos/lorenz.inverse.html
"""
import re
import numpy as np
import requests
import io
import matplotlib.pyplot as plt
import deepxde as dde

def gen_traindata():
    data = np.load("SIR_normalized.npz", allow_pickle = True)
    return data['arr_0'][:, 0:1], data['arr_0'][:, 1:4]



C1 = dde.Variable(1.0)
C2 = dde.Variable(1.0)


def SIR(x, y):
    """SIR system.
    dy1/dx = -beta*y1*y2
    dy2/dx = beta*y1*y2 - gamma*y2
    dy3/dx = gamma*y2
    """
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:]
    dy1_x = dde.grad.jacobian(y, x, i=0)
    dy2_x = dde.grad.jacobian(y, x, i=1)
    dy3_x = dde.grad.jacobian(y, x, i=2)
    return [
        dy1_x + C1 * (y2 * y1),
        dy2_x - C1 * (y2 * y1) + C2*y2,
        dy3_x - C2*y2,
    ]


def boundary(_, on_initial):
    return on_initial


geom = dde.geometry.TimeDomain(0, 600)

# Initial conditions
ic1 = dde.IC(geom, lambda X: -8, boundary, component=0)
ic2 = dde.IC(geom, lambda X: 7, boundary, component=1)
ic3 = dde.IC(geom, lambda X: 27, boundary, component=2)

# Get the train data
observe_t, ob_y = gen_traindata()
observe_y0 = dde.PointSetBC(observe_t, ob_y[:, 0:1], component=0)
observe_y1 = dde.PointSetBC(observe_t, ob_y[:, 1:2], component=1)
observe_y2 = dde.PointSetBC(observe_t, ob_y[:, 2:3], component=2)
'''
plt.plot(observe_t, ob_y)
plt.xlabel("Time")
plt.legend(["S", "I", "R"])
plt.title("Training data")
plt.show()
'''

data = dde.data.PDE(
    geom,
    SIR,
    [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
    num_domain=400,
    num_boundary=2,
    anchors=observe_t,
)

net = dde.maps.FNN([1] + [40] * 3 + [3], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile("adam", lr=0.001, external_trainable_variables=[C1, C2])
# callbacks for storing results
fnamevar = "variables.dat"
variable = dde.callbacks.VariableValue(
    [C1, C2], period=600, filename=fnamevar
)
losshistory, train_state = model.train(epochs=60000, callbacks=[variable])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
'''
# reopen saved data using callbacks in fnamevar
lines = open(fnamevar, "r").readlines()

# read output data in fnamevar (this line is a long story...)
Chat = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)

l, c = Chat.shape

C1true = 3/14
C2true = 1/14

plt.plot(range(l), Chat[:, 0], "r-")
plt.plot(range(l), Chat[:, 1], "k-")
#plt.plot(range(l), Chat[:, 2], "g-")
plt.plot(range(l), np.ones(Chat[:, 0].shape) * C1true, "r--")
plt.plot(range(l), np.ones(Chat[:, 1].shape) * C2true, "k--")
#plt.plot(range(l), np.ones(Chat[:, 2].shape) * C3true, "g--")
plt.legend(["C1hat", "C2hat", "True C1", "True C2"], loc="right")
plt.xlabel("Epoch")
plt.show()


yhat = model.predict(observe_t)

plt.plot(observe_t, ob_y, "-", observe_t, yhat, "--")
plt.ylim(0,1)
plt.xlabel("Time")
plt.legend(["S", "I", "R", "Sh", "Ih", "Rh"])
plt.title("Training data")
plt.show()
'''