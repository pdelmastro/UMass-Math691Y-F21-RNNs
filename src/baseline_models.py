"""
Basic RNN models used for baseline comparison
Branched from 'baseline_models.py' so previous experiments could still
    rely on the old code
"""
import numpy as np
import torch
import torch.nn as nn

"""
--------------------------------------------------------------------------------
    Basic fully connected RNN
--------------------------------------------------------------------------------
"""
class RNN(nn.Module):

    def __init__(self, N_in, N_h, N_out, N_m=0,p_uh=1, g_uh=1,
            p_hh=1, g_hh=1, tau=1, learn_tau=False, f_h='tanh', f_out='softmax',
            noise=0.0, seed=None
        ):
        """
        Network Size Parameters

            N_in        : Number of inputs to the network (i.e. input dimension)

            N_h         : Number of neurons in the hidden layer

            N_out       : Number of outputs of the network (i.e. output dimension)

        Structural Parameters

            TODO: add descriptions of p_ij, g_ij

        Time-constant parameters

            tau         : Used to set the initial time-constants of the network
                          If a float, all specifics the initial time-constant of
                            all neurons in the network
                          If (N_h,) numpy array, specifies the individual time
                            constants of each neuron
                          Default: float, 1

            learn_tau   : Flag. Set to true if tau values should be treated as
                          a network parameter and learned.

        Misc Parameters

            f_h         : Activation function of the hidden units.
                          Options are
                            'tanh' for hyperbolic tangent
                            'sigmoid' for logistic sigmoid
                            'softmax' for softmax
                            'id' for identity (i.e. no non-linearity)
                          Default: 'tanh'

            f_out       : Activation function of the output units.
                          Options are the same as those for parameter 'f_h'
                          Default: 'softmax'

            noise       : Stdev of noise injected in RNN update
                          Default: 0.0 (no noise)

            seed        : Int or None.
                          Random seed used when generating the network's parameters
        """
        super(RNN, self).__init__()

        # Save variable parameters
        self.N_in = N_in
        self.N_h = N_h
        self.N_out = N_out
        self.noise = noise


        # Non-linearities
        activations = []
        for f in [f_h, f_out]:
            if f == 'tanh':
                activations.append(nn.Tanh())
            elif f == 'sigmoid':
                activations.append(nn.Sigmoid())
            elif f == 'id':
                activations.append(nn.Identity())
            elif f == 'softmax':
                activations.append(nn.Softmax(dim=1))
            else:
                raise ValueError('Activation function %s not recognized' % f)
        self.f_h, self.f_out = activations
        self.f_h_type = f_h


        # Network weight matrices
        if seed is not None:
            np.random.seed(seed)


        # Input-hidden weights
        self.W_uh = nn.Parameter(sparse_normal_weights(N_in, N_h, p_uh, g_uh))

        # Hidden-X weights
        # Hidden-hidden weights
        self.W_hh = nn.Parameter(sparse_normal_weights(N_h, N_h, p_hh, g_hh))
        # Hidden-output weights
        self.W_hy = nn.Parameter(sparse_normal_weights(N_h, N_out))

        # Bias vectors
        self.b_h = nn.Parameter(
            torch.from_numpy(np.zeros(N_h).astype('float32'))
        )
        self.b_y = nn.Parameter(
            torch.from_numpy(np.zeros(N_out).astype('float32'))
        )

        # Hidden initial state
        if f_h != 'sigmoid':
            self.h0 = nn.Parameter(
                torch.from_numpy(2*np.random.random(size=(N_h)).astype('float32')-1)
            )
        else:
            self.h0 = nn.Parameter(
                torch.from_numpy(np.random.random(size=(N_h)).astype('float32'))
            )


        # Setup the time constants
        if isinstance(tau, float) or isinstance(tau, int):
            self.tau = nn.Parameter(tau * torch.ones(N_h))
        else:
            self.tau = nn.Parameter(torch.from_numpy(tau.astype('float32')))
        self.tau.requires_grad = learn_tau



    def forward(self, u, h0=None, return_dynamics=False, taus=None):
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

            taus    :   None or (# seq, ) Tensor or (# seq, self.N_h) Tensor
                        If None, uses the existing tau values of the network
                        If (# seqs,) Tensor, sets all neurons' tau values to
                            taus[i] when the network is run on u[i]
                        If (# seqs, self.N_h) Tensor, sets the j-th neuron's tau
                            value to tau[i,j] when the network is run on u[i]

                        Default: None
        """
        # Tensor to store the network output
        batch_size, T = u.shape[0], u.shape[1]

        if return_dynamics:
            hd = torch.zeros(batch_size, T, self.N_h)   # Network hidden state
        y = torch.zeros(batch_size, T, self.N_out)      # Network outputs

        # Tau values
        if taus is None:
            alpha = 1/self.tau
        # Different *global* tau value by example
        elif len(taus.shape) == 1:
            alpha = (1./taus.float()).repeat(self.N_h,1).t()
        # TODO: implement different tau values by example and neuron
        else:
            alpha = 1/taus.float()

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


    def clip_time_constants(self):
        """
        Clips the networks time-constants s.t. all tau >= 1
        """
        self.tau.data = torch.clamp(self.tau.data, min=1)


    def clip_initial_state(self):
        """
        Clips the networks initial state based on the network's activation type
        """
        if self.f_h_type == 'tanh':
            self.h0.data = torch.clamp(self.h0.data, min=-1, max=1)
        elif self.f_h_type == 'sigmoid':
            self.h0.data = torch.clamp(self.h0.data, min=0, max=1)

"""
--------------------------------------------------------------------------------
    Helper functions
--------------------------------------------------------------------------------
"""
def sparse_normal_weights(N_src, N_dst, p=1, g=1):
    """
    Generates an (N_dst x N_src)-array M where element i,j is generated
    as follows:
    - with probability 1-p, M[i,j] = 0
    - with probability p, M[i,j] with drawn from a normal distribution with
      zero mean and variance g^2 / (p * N_src)
    """
    # Edge case: p = 0
    if p == 0:
        M = np.zeros((N_dst, N_src))
    else:
        # Generate an (N_dst x N_src) matrix where all elements are drawn
        # from normal dist with zero mean and variance g^2/(p * N_src))
        M = np.random.normal(scale=(g/np.sqrt(p*max(1,N_src))), size=(N_dst, N_src))
        # Zero out elements with probability 1-p
        M *= np.random.choice([0,1], p=[1-p,p], size=(N_dst, N_src))
    # Return the matrix
    return torch.from_numpy(M.astype('float32')).t()


def excitatory_inhibitory_weights(N_exc, N_inh, N_dst, p=1, g=1,
        zero_diagonal=False
    ):
    """
    Generates an ( (N_exc + N_inh) x N_src)-array M where element i,j is generated
    as follows:

    1. Generate matrix M using the following process
       - with probability 1-p, M[i,j] = 0
       - with probability p, M[i,j] with drawn from a Gamma distribution with
         shape parameter k=2 and mean 1 if j < N_exc or mean N_exc / N_inh otherwise

    2. Rescale M s.t. Var[ sum_i M[i,j] ] is equal to g^2
    """
    # Useful constants
    N_src = N_exc + N_inh
    f_exc = N_exc / N_src
    mu_exc, mu_inh = 1, f_exc / (1 - f_exc)

    # Matrix generation
    M = np.zeros((N_dst, N_src))
    if p > 0:
        # Generate the matrix elements for each type of neuron
        # Excitatory
        M[:,:N_exc] = np.random.gamma(shape=2, scale=mu_exc/2, size=(N_dst, N_exc))
        # Inhibitory
        M[:,N_exc:] = np.random.gamma(shape=2, scale=mu_inh/2, size=(N_dst, N_inh))

        # Zero out elements with probability 1-p
        M *= np.random.choice([0,1], p=[1-p,p], size=(N_dst, N_src))

        # Remove diagonal connections
        if zero_diagonal and N_src == N_dst:
            M.fill_diagonal(0)

        # Rescale to get the correct variance
        v = 0.5 * p * N_src * (f_exc + (1-f_exc) * (mu_inh ** 2))
        M *= (g / np.sqrt(v))

        # if N_dst == N_src:
        #     signs = np.ones(N_src)
        #     signs[N_exc:] = -1
        #     D = np.diag(signs)
        #     print(np.max(np.abs(np.linalg.eigvals(M @ D))))

    # Return the matrix
    return torch.from_numpy(M.astype('float32')).t()
