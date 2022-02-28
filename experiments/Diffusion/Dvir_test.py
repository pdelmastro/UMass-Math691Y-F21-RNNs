import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import matplotlib.gridspec as gridspec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#class PINN(nn.Module):
#    def __init__(self):
#        def __init__(self):
#            super(PINN, self).__init__()
#            self.sequential_liner = nn.Sequential(
#                nn.Linear(2, 100),
#                nn.Tanh(),
#                nn.Linear(100, 1)
#            )
#
#        def forward(self, x):
#            x = x.view(len(x), -1)
#            return self.sequential_liner(x)



class PINN(nn.Module):
    def __init__(self,layers, nodes):
        super(PINN, self).__init__()
        self.input_layer = nn.Linear(2,nodes)
        self.hidden_layer = nn.ModuleList([nn.Linear(nodes,nodes) for i in range(layers)])
        self.output_layer = nn.Linear(nodes,1)

    def forward(self,x):
        outcome = torch.tanh(self.input_layer(x))
        for i ,li in enumerate(self.hidden_layer):
            outcome = torch.tanh(li(outcome))
        output = self.output_layer(outcome)
        return output
# here i am using tanh as the activation function.
# change the line nn.Tanh() can switch to other activation function.


def initialization(sample_size):
    x = torch.cat((torch.rand([sample_size,1]),torch.rand([sample_size,1])),1)
    # x = (x,t)
    u_initial = torch.cat((torch.rand([sample_size,1]),torch.zeros([sample_size,1])),1)
    # u(x,0)
    u_boundary_left = torch.cat((torch.zeros([sample_size,1]),torch.rand([sample_size,1])),1)
    #u(0,t)
    u_boundary_right = torch.cat((torch.ones([sample_size,1]),torch.rand([sample_size,1])),1)
    #u(1,t)
    return x,u_initial,u_boundary_left,u_boundary_right


# the initial condition is u(x,0) = sin(2*pi*x)
# u(0,t) = u(1,t) = 0
# x in [0,1]
# t in [0,1]
# the heat function is du/dt = k*du^2/dx^2



def main():
    iteration = 10000
    sample_size = 200
    lr = 0.001
    x_train,u_initial,u_boundary_left,u_boundary_right = initialization(sample_size)
    net = PINN(2,100).to(device)
    optimizer = optim.Adam(net.parameters())
    loss_fun = nn.MSELoss(reduce='mean')
    


  # training
    for i in range(iteration):
        optimizer.zero_grad()


        #send variables to GPU
        x_train = Variable(x_train,requires_grad = True).to(device)
        u_initial = Variable(u_initial).to(device)
        u_boundary_left = Variable(u_boundary_left).to(device)
        u_boundary_right = Variable(u_boundary_right).to(device)

        du = torch.autograd.grad(net(x_train), x_train, grad_outputs=torch.ones_like(net(x_train)), create_graph=True)
        ux = du[0][:,0].unsqueeze(-1)
        ut = du[0][:,1].unsqueeze(-1)
        uxx = torch.autograd.grad(ux, x_train, grad_outputs=torch.ones_like(ux), create_graph=True)[0][:,0].unsqueeze(-1)
        #y_train = net(x_train)


        #loss function:
        loss1 = loss_fun(net(u_initial),torch.sin(2*np.pi*x_train))
        # computing the loss of u(x,0)

        loss2 = loss_fun(net(u_boundary_left),torch.zeros(sample_size,1))
        #computing the loss of u(0,t)

        loss3 = loss_fun(net(u_boundary_right),torch.zeros([sample_size,1]))
        #computing the loss of u(1,t)

        loss4 = loss_fun(ut-uxx,torch.zeros([sample_size,1]))

        loss = loss1+loss2+loss3+loss4
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print(f'step: {i} loss = {loss.item()}')

            # plotting
            fig = plt.figure()
            G = gridspec.GridSpec(3, 3)
            ax1 = fig.add_subplot(G[0:2,0], projection='3d')
            ax2 = fig.add_subplot(G[0:2,1], projection='3d')
            ax3 = fig.add_subplot(G[0:2,2], projection='3d')
            x = np.linspace(0,1,sample_size)
            t = np.linspace(0,1,sample_size)
            temp = np.empty((2,1))
            pred = np.zeros((sample_size,sample_size))
            for i in range(sample_size):
                temp[0] = x[i]
                for j in range(sample_size):
                    temp[1] = t[j]
                    ctemp = torch.Tensor(temp.reshape(1,-1))
                    pred[i][j] = net(ctemp).detach().numpy()
            X,T = np.meshgrid(x,t,indexing = 'ij')
            pred = np.reshape(pred,(t.shape[0],x.shape[0]))
            u = np.sin(2*np.pi*X)*np.exp((-4*np.pi**2)*T)
            ax1.plot_surface(X,T,pred)
            ax2.plot_surface(X,T,u)
            ax2.plot_surface(X,T,u-pred)
            ax1.set_xlabel('x')
            ax1.set_ylabel('t')
            ax1.set_zlabel('u')
            ax2.set_xlabel('x')
            ax2.set_ylabel('t')
            ax2.set_zlabel('u')
            ax3.set_xlabel('x')
            ax3.set_ylabel('t')
            ax3.set_zlabel('u')
            plt.title('Network Solution (left), True Solution (center), Difference of Solutions (right)')
            plt.show()

    torch.save(net.state_dict(),'1D_heat_equation_Dvir.pt')
if __name__ == '__main__':
    main()

# the exact solution is u = sin(2*pi*x)e^(-4*pi**2)t