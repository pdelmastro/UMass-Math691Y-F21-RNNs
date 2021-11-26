"""
Basic Gated RNN models used for baseline comparison

TODO: upgrade to same format as baseline_models.py
"""
import numpy as np
import torch
import torch.nn as nn
from .baseline_models import sparse_normal_weights

"""
--------------------------------------------------------------------------------
    Basic fully connected GRU
--------------------------------------------------------------------------------
"""
class GRU(nn.Module):

    def __init__(self, N_in, N_h, N_out, p_uh=1, g_uh=1, p_hh=1, g_hh=1,
            g_mh=1, p_mh=1, g_hm=1, p_hm=1, g_mm=1, p_mm=1,
            f_out='softmax', seed=None
        ):
        """
        RNN Constructor

        Network Size Parameters

            N_in        : Number of inputs to the network (i.e. input dimension)

            N_h         : Number of neurons in the hidden layer

            N_out       : Number of outputs of the network (i.e. output dimension)z

            N_m         : Number of neurons in the working memory layer
        """
        super(GRU, self).__init__()

        # Save the dimensions of the network
        self.N_in = N_in
        self.N_h = N_h
        self.N_out = N_out

        # Non-linearities
        self.f_h = nn.Tanh()
        self.f_g = nn.Sigmoid()
        self.f_out = nn.Softmax(dim=1)
        if f_out == 'softmax':
            self.f_out = nn.Softmax(dim=1)
        elif f_out == 'tanh':
            self.f_out = nn.Tanh()
        elif f_out == 'sigmoid':
            self.f_out = nn.Sigmoid()
        else:
            raise ValueError('Output non-linearity %s not recognized' % f_out)

        # Network weight matrices
        if seed is not None:
            np.random.seed(seed)
        # Input-hidden weights
        self.W_uz = nn.Parameter(sparse_normal_weights(N_in, N_h))
        self.W_ur = nn.Parameter(sparse_normal_weights(N_in, N_h))
        self.W_uh = nn.Parameter(sparse_normal_weights(N_in, N_h, p_uh, g_uh))
        # Hidden-hidden weights
        self.W_hz = nn.Parameter(sparse_normal_weights(N_h, N_h))
        self.W_hr = nn.Parameter(sparse_normal_weights(N_h, N_h))
        self.W_hh = nn.Parameter(sparse_normal_weights(N_h, N_h, p_hh, g_hh))
        # Hidden-output weights
        self.W_hy = nn.Parameter(sparse_normal_weights(N_h, N_out))


        # Bias vectors
        self.b_z = nn.Parameter(
            torch.from_numpy(np.zeros(N_h).astype('float32'))
        )
        self.b_r = nn.Parameter(
            torch.from_numpy(np.zeros(N_h).astype('float32'))
        )
        self.b_h = nn.Parameter(
            torch.from_numpy(np.zeros(N_h).astype('float32'))
        )
        self.b_y = nn.Parameter(
            torch.from_numpy(np.zeros(N_out).astype('float32'))
        )

        # Hidden initial state
        self.h0 = nn.Parameter(
            torch.from_numpy(np.random.normal(size=(N_h)).astype('float32'))
        )


    def forward(self, u, return_dynamics=False):
        # Tensor to store the network output
        batch_size, T = u.shape[0], u.shape[1]

        # Arrays to store network dynamics
        y = torch.zeros(batch_size, T, self.N_out)  # Network outputs
        if return_dynamics:
            hd = torch.zeros(batch_size, T, self.N_h)
            upd = torch.zeros(batch_size, T, self.N_h)
            res = torch.zeros(batch_size, T, self.N_h)

        # Loop over time to compute network dynamics
        h = self.h0.repeat(batch_size,1)
        for t in range(T):
            # Update and reset gates
            z = self.f_g(
                h.matmul(self.W_hz) + u[:,t,:].matmul(self.W_uz) + self.b_z
            )
            r = self.f_g(
                h.matmul(self.W_hr) + u[:,t,:].matmul(self.W_ur) + self.b_r
            )
            # Hidden state update
            h_tilde = self.f_h(
                (r * h).matmul(self.W_hh) + u[:,t,:].matmul(self.W_uh) + self.b_h
            )
            h = (1-z) * h + z * h_tilde

            # Output
            y[:,t,:] = self.f_out(h.matmul(self.W_hy) + self.b_y)

            # Save dynamics
            if return_dynamics:
                hd[:,t,:] = h
                upd[:,t,:] = z
                res[:,t,:] = r

        # Return the generated dynamics and/or output
        if return_dynamics:
            return hd, upd, res, y
        else:
            return y


"""
--------------------------------------------------------------------------------
    Basic fully connected LSTM
--------------------------------------------------------------------------------
"""
class LSTM(nn.Module):

    def __init__(self, N_in, N_h, N_out, p_uh=1, g_uh=1, p_hh=1, g_hh=1,
            g_mh=1, p_mh=1, g_hm=1, p_hm=1, g_mm=1, p_mm=1,
            f_out='softmax', seed=None
        ):
        """
        RNN Constructor

        Network Size Parameters

            N_in        : Number of inputs to the network (i.e. input dimension)

            N_h         : Number of neurons in the hidden layer

            N_out       : Number of outputs of the network (i.e. output dimension)

            N_m         : Number of neurons in the working memory layer
        """
        super(LSTM, self).__init__()

        # Save the dimensions of the network
        self.N_in = N_in
        self.N_h = N_h
        self.N_out = N_out

        # Non-linearities
        self.f_h = nn.Tanh()
        self.f_g = nn.Sigmoid()
        self.f_out = nn.Softmax(dim=1)
        if f_out == 'softmax':
            self.f_out = nn.Softmax(dim=1)
        elif f_out == 'tanh':
            self.f_out = nn.Tanh()
        elif f_out == 'sigmoid':
            self.f_out = nn.Sigmoid()
        else:
            raise ValueError('Output non-linearity %s not recognized' % f_out)

        # Network weight matrices
        if seed is not None:
            np.random.seed(seed)
        # Input-hidden weights
        self.W_uf = nn.Parameter(sparse_normal_weights(N_in, N_h))
        self.W_ui = nn.Parameter(sparse_normal_weights(N_in, N_h))
        self.W_uo = nn.Parameter(sparse_normal_weights(N_in, N_h))
        self.W_uc = nn.Parameter(sparse_normal_weights(N_in, N_h))
        self.W_uh = nn.Parameter(sparse_normal_weights(N_in, N_h))
        # Hidden-hidden weights
        self.W_hf = nn.Parameter(sparse_normal_weights(N_h, N_h))
        self.W_hi = nn.Parameter(sparse_normal_weights(N_h, N_h))
        self.W_ho = nn.Parameter(sparse_normal_weights(N_h, N_h))
        self.W_hc = nn.Parameter(sparse_normal_weights(N_h, N_h))

        # Hidden-output weights
        self.W_hy = nn.Parameter(sparse_normal_weights(N_h, N_out))

        # Bias vectors
        self.b_f = nn.Parameter(
            torch.from_numpy(np.zeros(N_h).astype('float32'))
        )
        self.b_i = nn.Parameter(
            torch.from_numpy(np.zeros(N_h).astype('float32'))
        )
        self.b_o = nn.Parameter(
            torch.from_numpy(np.zeros(N_h).astype('float32'))
        )
        self.b_c = nn.Parameter(
            torch.from_numpy(np.zeros(N_h).astype('float32'))
        )
        self.b_y = nn.Parameter(
            torch.from_numpy(np.zeros(N_out).astype('float32'))
        )

        # Hidden initial state
        self.h0 = nn.Parameter(
            torch.from_numpy(np.random.normal(size=(N_h)).astype('float32'))
        )
        self.c0 = nn.Parameter(
            torch.from_numpy(np.random.normal(size=(N_h)).astype('float32'))
        )


    def forward(self, u, return_dynamics=False):
        # Tensor to store the network output
        batch_size, T = u.shape[0], u.shape[1]

        # Arrays to store network dynamics
        y = torch.zeros(batch_size, T, self.N_out)  # Network outputs
        if return_dynamics:
            hd = torch.zeros(batch_size, T, self.N_h)
            cell = torch.zeros(batch_size, T, self.N_h)
            # TODO: add forget, input, and output gates

        # Loop over time to compute network dynamics
        h = self.h0.repeat(batch_size,1)
        c = self.c0.repeat(batch_size,1)
        for t in range(T):
            # Forget, input, and out gates
            f = self.f_g(
                h.matmul(self.W_hf) + u[:,t,:].matmul(self.W_uf) + self.b_f
            )
            i = self.f_g(
                h.matmul(self.W_hi) + u[:,t,:].matmul(self.W_ui) + self.b_i
            )
            o = self.f_g(
                h.matmul(self.W_ho) + u[:,t,:].matmul(self.W_uo) + self.b_o
            )
            # Hidden state update
            c_tilde = self.f_h(
                h.matmul(self.W_hc) + u[:,t,:].matmul(self.W_uc) + self.b_c
            )
            c = f * h + i * c_tilde
            h = o * self.f_h(c)

            # Output
            y[:,t,:] = self.f_out(h.matmul(self.W_hy) + self.b_y)

            # Save dynamics
            if return_dynamics:
                hd[:,t,:] = h
                cell[:,t,:] = c

        # Return the generated dynamics and/or output
        if return_dynamics:
            return hd, cell, y
        else:
            return y
