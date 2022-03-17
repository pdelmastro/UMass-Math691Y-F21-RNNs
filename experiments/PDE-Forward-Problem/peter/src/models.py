
import torch
import torch.nn as nn

class FCNN(nn.Module):
    """
    Fully connected feed-forward neural network
    """
    def __init__(self, N_h, N_in=2, N_out=1, activation='tanh'):
        super().__init__()

        # Save network parameters
        self.N_in = N_in
        self.N_out = N_out

        # Hidden layer size may be int or list of ints
        if isinstance(N_h, int):
            self.N_h = [N_h]
        else:
            self.N_h = N_h
        self.N_layers = len(self.N_h)

        # Select the activation function
        if activation == 'tanh':
            activation = nn.Tanh
        elif activation == 'relu':
            activation = nn.ReLU
        elif activation == 'sine':
            activation = torch.sin

        # Create the layers
        self.fcs = nn.Sequential(*[
            nn.Linear(self.N_in, self.N_h[0]),
            activation()
        ])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(self.N_h[i], self.N_h[i+1]),
                activation()
            ])
            for i in range(self.N_layers-1)]
        )
        self.fce = nn.Linear(self.N_h[-1], self.N_out)


    def forward(self, v):
        nn_out = self.fcs(v)
        nn_out = self.fch(nn_out)
        nn_out = self.fce(nn_out)
        return nn_out
