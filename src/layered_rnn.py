
import numpy as np
import torch
import torch.nn as nn

"""
--------------------------------------------------------------------------------
    LayeredRNN Class

This might be useful later in the project if we decide to try out some more 
powerful RNN architectures
--------------------------------------------------------------------------------
"""

class LayeredRNN(nn.Module):

    def __init__(self, 
            N_in, N_h, N_out, 
            P_uh=1, G_uh=1, G_hh=1, G_hh=1, P_hy=1, G_hy=1, 
            layer_params=None, 
            noise=0.0, seed=None
        ):
        """
        Network Size Parameters

            N_in        : Int. Number of inputs to the network (i.e. input dimension)

            N_h         : List. Number of neurons in each hidden layer

            N_out       : Int. Number of outputs of the network (i.e. output dimension)

        Structural Parameters

            TODO: add descriptions of P_ij, G_ij
            TODO: decide on the default parameters for these 

            layer_params    :   List. 
                                TODO: explain. Basically this is the kwargs
                                for the RNNCell of each layer

        Misc Parameters

            noise       : Stdev of noise injected in RNN update
                          Default: 0.0 (no noise)

            seed        : Int or None.
                          Random seed used when generating the network's parameters
        """
        super(LayeredRNN, self).__init__()

        """
        Basic Setup
        """

        # Save variable parameters
        self.N_in = N_in
        self.N_h = N_h
        self.N_out = N_out
        self.noise = noise
        self.layer_params = layer_params

        # Determine the number of layers in the network
        if isinstance(N_h, int):
            self.N_layers = 1
        else:
            self.N_layers = len(N_h)

        """
        Creating the layers
        """
        self.layers = []
        for param_dict in layer_params:
            # Determine the correct constructor for the layer
            cell_type = param_dict['layer_type']
            # TODO: implement

            # Construct the layer and add it to the list of 
            # layers for this network
            # Something like the following:
            # self.layers.append(
            #     cell_constructor(**param_dict)
            # )

        """
        Inter-layer weights
        """
        # Handle the parameters P_xy and G_xy

        # Now build the weights
        self.W_hh = {}
        for i in range(self.N_layers):
            for j in range(self.N_layers):
                pass


    def forward(self, u, h0=None, return_dynamics=False):
        """
        ARGUMENTS

            u       :   (# seqs, seq length, self.N_in) Tensor
                        storing a batches of input sequences for the network

            h0      :   None or (self.N_h,) Tensor or (# seqsm self.N_h) Tensor
                        If None, the RNN's initial hidden state is self.h0
                            for all input sequences u[i]
                        If (self.N_h,) Tensor, this will be the RNN's initial
                            hidden state for all input sequences u[i]
                        If (# seqs, self.N_h) Tensor, the RNN's initial hidden
                            state for input u[i] will be h0[i]

            return_dynamics     :   Flag. Set to True for this function to
                                    return all the networks sequences of hidden
                                    states in response to this given input
        """
        # Tensor to store the network output
        batch_size, T = u.shape[0], u.shape[1]

        if return_dynamics:
            hd = torch.zeros(batch_size, T, self.N_h)   # Network hidden state
        y = torch.zeros(batch_size, T, self.N_out)      # Network outputs

        # Initial state
        if h0 is None:
            self.clip_initial_state()
        else:
            # TODO: implement
            pass
        h = self.h0 if h0 is None else h0

        # Loop over time to compute network dynamics
        for t in range(T):
            # Hidden state update
            u_h = h.matmul(self.W_hh) + u[:,t,:].matmul(self.W_uh) \
                  + self.b_h + self.noise * torch.randn(h.shape)
            h = (1 - alpha) * h + alpha * self.f_h(u_h)

            # Output
            y[:,t,:] = self.f_out(h.matmul(self.W_hy) + self.b_y)

            # Save the hidden state
            if return_dynamics:
                hd[:,t,:] = h

        # Return the generated memories and output (and possibly hidden states)
        if return_dynamics:
            return hd, y
        else:
            return y