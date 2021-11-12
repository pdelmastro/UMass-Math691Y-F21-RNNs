"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import pandas as pd

us_cases_df = pd.read_csv("US_Covid_Cases.csv")
cases_series = us_cases_df.loc[:,"cases"]

# Training data will be first 80% of data and Test will be last 20% of data
n = 80
m = 20
fname_train = cases_series.head(int(len(cases_series)*(n/100)))
fname_test = cases_series.tail(int(len(cases_series)*(m/100)))
data = dde.data.DataSet(
    fname_train=fname_train,
    fname_test=fname_test,
    col_x=(0,),
    col_y=(1,),
    #standardize=True,
)

layer_size = [1] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot normal"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(epochs=50000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)