import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import requests
import io
import deepxde as dde

df = pd.read_csv('SIR.csv')

df = df.head(327)
us_pop = 333000000

immediate_cases = []
for i in range(len(df['Cases [C]'])):
    if i is 0:
        continue
    immediate_cases.append(df['Cases [C]'].iloc[i]-df['Cases [C]'].iloc[i-1])

df = df.head(326)
df['I Vals'] = pd.Series(immediate_cases)


df['Susceptible'] = us_pop - df['Recovered [R]'] - df['I Vals']
days_list = list(range(0,len(df['Susceptible'])))
df['Days'] = pd.Series(days_list)
df['S'] = df['Susceptible']/us_pop
df['I'] = df['I Vals']/us_pop
df['R'] = df['Recovered [R]']/us_pop

# Let's focus on one peak from day 50 to day 150
df_sub = df.iloc[50:150]
days_list = []
s_list = []
i_list = []
r_list = []
for i in range(len(df_sub)):
    days_list.append(df['Days'].iloc[i+50])
    s_list.append(df['S'].iloc[i])
    i_list.append(df['I'].iloc[i])
    r_list.append(df['R'].iloc[i])

new_df = pd.DataFrame(list(zip(days_list,s_list,i_list,r_list)), 
    columns=['Days', 'S', 'I', 'R'])
t_array = new_df['Days'].to_numpy()
y_array = new_df[['S','I','R']].to_numpy()
concat = np.column_stack([t_array, y_array])

def gen_traindata():
    return concat[:,0:1], concat[:,1:4]

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


geom = dde.geometry.TimeDomain(50, 150)

# Initial conditions
ic1 = dde.IC(geom, lambda X: float(new_df['S'][0]), boundary, component=0)
ic2 = dde.IC(geom, lambda X: float(new_df['I'][0]), boundary, component=1)
ic3 = dde.IC(geom, lambda X: float(new_df['R'][0]), boundary, component=2)

# Get the train data
observe_t, ob_y = gen_traindata()
observe_y0 = dde.PointSetBC(observe_t, ob_y[:, 0:1], component=0)
observe_y1 = dde.PointSetBC(observe_t, ob_y[:, 1:2], component=1)
observe_y2 = dde.PointSetBC(observe_t, ob_y[:, 2:3], component=2)

data = dde.data.PDE(
    geom,
    SIR,
    [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
    num_domain=600,
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
#dde.saveplot(losshistory, train_state, issave=True, isplot=True)

yhat = model.predict(observe_t)

plt.plot(observe_t, ob_y[:, 1:2], "-", observe_t, yhat[:, 1:2], "--")
plt.xlabel("Time")
plt.legend(["I","Ih"])
plt.title("Real I vs Training data I for SIR")
plt.show()
