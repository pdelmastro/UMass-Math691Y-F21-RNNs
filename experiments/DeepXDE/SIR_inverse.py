"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch
Documentation: https://deepxde.readthedocs.io/en/latest/demos/lorenz.inverse.html
"""
import deepxde as dde
import numpy as np
import pandas as pd


def gen_traindata():
    data = pd.read_csv('SIR_dummy_data.csv')
    data['t'] = data['t'].astype(int)
    data['S'] = data['S'].mul(325000000)
    data['I'] = data['I'].mul(325000000)
    data['R'] = data['R'].mul(325000000)
    data['S'] = data['S'].astype(int)
    data['I'] = data['I'].astype(int)
    data['R'] = data['R'].astype(int)
    t_array = data['t'].to_numpy()
    y_array = data[['S', 'I', 'R']].to_numpy()
    return t_array, y_array


'''
def gen_traindata():
    data = np.load("Lorenz.npz")
    return data["t"], data["y"]
'''


C1 = dde.Variable(1.0)
C2 = dde.Variable(1.0)
#C3 = dde.Variable(1.0)


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


geom = dde.geometry.TimeDomain(0, 3)

# Initial conditions
ic1 = dde.IC(geom, lambda X: -8, boundary, component=0)
ic2 = dde.IC(geom, lambda X: 7, boundary, component=1)
ic3 = dde.IC(geom, lambda X: 27, boundary, component=2)

# Get the train data
observe_t, ob_y = gen_traindata()
observe_y0 = dde.PointSetBC(observe_t, ob_y[:, 0:1], component=0)
observe_y1 = dde.PointSetBC(observe_t, ob_y[:, 1:2], component=1)
observe_y2 = dde.PointSetBC(observe_t, ob_y[:, 2:3], component=2)
print(ob_y)
'''
data = dde.data.PDE(
    geom,
    SIR,
    [ic1, ic2, observe_y0, observe_y1, observe_y2],
    num_domain=400,
    num_boundary=2,
    anchors=observe_t,
)

net = dde.maps.FNN([1] + [40] * 3 + [3], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile("adam", lr=0.001, external_trainable_variables=[C1, C2])
variable = dde.callbacks.VariableValue(
    [C1, C2], period=600, filename="variables.dat"
)
losshistory, train_state = model.train(epochs=60000, callbacks=[variable])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
'''