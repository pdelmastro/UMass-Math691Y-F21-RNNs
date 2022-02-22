import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np



class PINN(nn.Module):
    def __init__(self,layers, nodes):
        super(PINN, self).__init__()
        self.input_layer = nn.Linear(1,nodes)
        self.hidden_layer = nn.ModuleList([nn.Linear(nodes,nodes) for i in range(layers)])
        self.output_layer = nn.Linear(nodes,1)

    def forward(self,x):
        outcome = torch.tanh(self.input_layer(x))
        for i ,li in enumerate(self.hidden_layer):
            outcome = torch.tanh(li(outcome))
        output = self.output_layer(outcome)
        return output
#   u' = u
def main():
    x = torch.linspace(0,1,2000,requires_grad=True).unsqueeze(-1)
    # the range of x will cause a crucial problem 2 is recomended
    f = torch.tensor(torch.exp(x),requires_grad=True)
    net = PINN(3,50)
    lr = 0.0005
    MSE_cost_function = nn.MSELoss(reduce='mean')
    optimizer = torch.optim.Adam(net.parameters(),lr)
    iteration = 10000
    for epoch in range(iteration):
        optimizer.zero_grad()
        f_0 = net(torch.zeros(1))
        ux = torch.autograd.grad(net(x).sum(),x,create_graph=True)[0]
        y_train = net(x)
        MSE1 = MSE_cost_function(y_train,ux)
        MSE2 = MSE_cost_function(f_0,torch.ones(1))
        loss = MSE1 + MSE2
        if epoch % 100 == 0:
            print(y_train)
            plt.figure(1)
            plt.plot(x.detach().numpy(),f.detach().numpy(),'r',label  = 'True solution')
            plt.plot(x.detach().numpy(),y_train.detach().numpy(),'b',lw=2,label = 'approximation')
            plt.legend()
            plt.pause(.5)
            plt.show()
        loss.backward()
        optimizer.step()



if __name__ == '__main__':
    main()