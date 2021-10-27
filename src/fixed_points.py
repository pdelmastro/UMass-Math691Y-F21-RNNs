"""
Methods for finding fixed points of RNNs
Based on "Opening the Black Box: Low-Dimensional Dynamicsin High-Dimensional
Recurrent Neural Networks" by D. Sussillo and O. Barak (2013)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.functional import jacobian

def find_fixed_points(network, init_states, dh=1e-3, dh_decay=1,
        iters_before_first_dh_update=1e4, clip_gradient_thres=None,
        tol=1e-5, max_iters=1e4, return_jacobians=False, verbose=False
    ):
    """
    ARGUMENTS

        network         :   RNN whose fixed points will be computed

        init_states     :   (# states, # hidden units) PyTorch Tensor, w/
                            Initial hidden states used to computing the
                            fixed points

        dh              :   Float. Step-size for fixed point algorithm

        dh_decay        :   Float in [0,1]. Exponential decay rate of the
                            step size 'dh'

        iters_before_first_dh_update    : Int. Number of iterations before
                                          the step size 'dh' is updated for
                                          the first time

        clip_gradient_thres :   Gradients clipping threshold, under L_inf norm.
                                Set to 'None' for no clipping
                                Default: None

        tol             :   Float. Convergence tolerance threshold

        max_iters       :   Int. Max number of iterations allowed before
                            the fixed point algorithm terminates

        return_jacobians:   Bool. Set to True to return the Jacobian matrix
                            at each fixed point returned by this method

        verbose         :   Bool. Set to True to print out progress updates
                            as the algorithm runs

    RETURNS

        fixed_points    :   (# fixed points, # hidden units) PyTorch Tensor,
                            storing the fixed points computed by this function

        J               :   (# fixed points, # hidden units, # hidden units)
                            PyTorch Tensor, storing the Jacobian matrix at
                            each fixed point.
                            *** This is only returned when argument
                            'return_jacobians' is set to true

    """
    # Useful constants
    N_in = network.N_in
    n_states, N_h = init_states.shape

    # Freeze the network's parameters
    for param in network.parameters():
        param.requires_grad = False

    # Variable to be optimized, storing the initial hidden states
    h = torch.clone(init_states)
    h.requires_grad = True

    # Setup the optimizer and lr scheduler
    optimizer = torch.optim.Adam([h], lr=dh)
    dh_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=optimizer, gamma=dh_decay
                )
    n_dh_updates = 0
    iters_til_dh_update = iters_before_first_dh_update

    dh_traj = np.zeros((max_iters,n_states))

    # Begin the optimization loop
    for i in range(max_iters):

        # Step each hidden state forward in time once
        h_next = network(
            torch.zeros(n_states,1,N_in), h0=h, return_dynamics=True
        )[0].view(-1,N_h)

        # Check for convergence
        with torch.no_grad():
            dh_max = torch.max(torch.abs(h_next-h))
            dh_minmax = torch.abs(h_next-h).max(dim=1)[0].min()
            dh_traj[i] = torch.abs(h_next-h).max(dim=1)[0].detach().numpy()
        if dh_max < tol:
            break

        # Progress update
        if verbose and (i+1) % 64 == 0:
            print('Iter {}, Max |dh|: {:.6f}, Minmax |dh|: {:.6f}'.format(
                    i+1,dh_max, dh_minmax
                )
            )

        # Now compute the mean velocity across all examples
        q = 0.5 * torch.mean(torch.square(h_next-h))

        # Compute gradients
        q.backward()

        # Clip gradients for TFA weights
        if clip_gradient_thres is not None:
            nn.utils.clip_grad_norm_(
                [h], clip_gradient_thres
            )

        # Update the fixed point candidates
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()

        # dh update
        # Update the learning rate
        if iters_til_dh_update == 0:
            dh_scheduler.step()
            n_dh_updates += 1
            iters_til_dh_update = iters_before_first_dh_update * (2 ** n_dh_updates)
        else:
            iters_til_dh_update -= 1

    # if i+1 == max_iters:
    #     print('Algorithm failed to converge')

    # Testing
    # from sklearn.decomposition import PCA
    # import matplotlib.pyplot as plt
    # from mpl_toolkits import mplot3d
    #
    # h2 = h_traj.reshape(-1, N_h)
    # h2 -= np.mean(h2, axis=0)
    # n_components=3
    # pca = PCA(n_components=n_components)
    # h_pca = pca.fit_transform(h2)
    # h_pca = h_pca.reshape(max_iters,-1,n_components)
    #
    # plt.figure(figsize=(8,8))
    # ax = plt.axes(projection='3d')
    # for i in [112]:#, 56, 3,6,32]:
    #     ax.plot3D(h_pca[:,i,0], h_pca[:,i,1], h_pca[:,i,2], linewidth=1)
    # ax.set_xlabel('PCA 0')
    # ax.set_ylabel('PCA 1')
    # ax.set_zlabel('PCA 2')
    # plt.show()

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12,6))
    # plt.imshow(dh_traj[int(max_iters/2):].T, interpolation='none', aspect='auto')
    # plt.show()
    # exit()




    # Filter out duplicate fixed points
    fixed_points = unique_fixed_points(h.detach())

    # Compute the Jacobian matrices if requested
    if return_jacobians:
        return fixed_points, compute_jacobian(network, fixed_points)
    # Else, finished!
    return fixed_points


def unique_fixed_points(candidates, atol=5e-02, rtol=1e-06):
    """
    Removes duplicate candidate fixed points.
    Two points h1, h2 are considered identical if

            |h1[i] - h2[i]| < atol + |h2[i]|

    is satisfied for all indices i
    """
    unique_inds = []

    # loop over the candidates for unique fps
    for c in range(candidates.shape[0]):
        # compare to the existing set of fps
        unique = True
        for i in unique_inds:
            if candidates[c].allclose(candidates[i], rtol=rtol, atol=atol):
                unique = False
                break
        if unique: unique_inds.append(c)

    # Filter out duplicates now
    unique_fps = candidates[np.array(unique_inds)]

    # Done!
    return unique_fps


def compute_jacobian(network, hd):
    """
    Computes the Jacobian matrices for each hidden state hd[i] of the network
    """
    J = torch.empty(hd.shape[0], hd.shape[1], hd.shape[1])
    def autonomous_update(h):
        return network(
            torch.zeros(1,1,network.N_in),
            h0=h.view(-1,network.N_h),
            return_dynamics=True
        )[0].view(-1,network.N_h)
    for k in range(hd.shape[0]):
        J[k] = jacobian(autonomous_update, hd[k])
    return J.detach().numpy()
