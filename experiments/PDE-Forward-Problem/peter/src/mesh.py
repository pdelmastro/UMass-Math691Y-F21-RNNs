"""
Functions working with regular meshes on [0,1]^d x [0,T]
"""

import torch.nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
--------------------------------------------------------------------------------
    Regular Mesh
--------------------------------------------------------------------------------
"""
def generate_mesh(nt, nx, d=1, T=1):
    """
    Function to generate mesh points on [0,1]^d x [0,T]

     ARGUMENTS

        nt, nx   :   Ints. Number of temporal and spatial interior points
                     for the mesh, respectively

        d        :   Int. Spatial dimension
                     Default: 1
                     Note: Only works for d = 1 currently

        T        :   Float. Sets time domain to [0,T]
                     Default: 1

    RETURNS

        mesh_interior     :   (nt * nx^d, d+1)- Torch array of the interior of
                                the spatiotemporal mesh (t != 0 and x_i \notin {0,1}
                                for all i)

        mesh_boundary     :   (???, d+1) - Torch array of the spatial boundary
                                of the spatiotemporal mesh (x_i = 0 or x_i = 1
                                for some i = 1, ..., d)

        mesh_t0           :   (nx*d, d+1)- Torch array of the t=0 mesh points


    NOTE: Currently only works with d=1.
    TODO: Generalize to d > 1
    """
    t_vals = torch.linspace(0,T,steps=nt+2)
    x_vals = torch.linspace(0,1,steps=nx+2)

    # Create the interior mesh using meshgrid
    mesh = torch.meshgrid(
        [t_vals] + [x_vals] * d, indexing='ij'
    )

    # Break out the boundary, interior and t=0
    # Interior
    mesh_interior = torch.vstack([
        mesh_dim[1:,1:-1].ravel()
        for mesh_dim in mesh
    ])
    # Boundary
    # Left boundary
    left = torch.vstack([
        mesh_dim[1:,0]
        for mesh_dim in mesh
    ])
    # Right boundary
    right = torch.vstack([
        mesh_dim[1:,-1]
        for mesh_dim in mesh
    ])
    # Combined
    mesh_boundary = torch.hstack((left,right))

    # t = 0
    mesh_t0 = torch.vstack([
        mesh_dim[0,:]
        for mesh_dim in mesh
    ])

    # Return
    return mesh_interior.t(), mesh_boundary.t(), mesh_t0.t()


"""
--------------------------------------------------------------------------------
    Misc Functions
--------------------------------------------------------------------------------
"""
def reconstruct_u_array(u_interior, u_bndry, u_t0):
    """
    Function to reconstruct the (n_t+2, n_x+2) solution given the solution at
    t=0, on the boundaries, and inside the mesh
    """
    # Determine n_t and n_x
    n_x = u_t0.shape[0] - 2
    n_t = int(u_interior.shape[0] / n_x) - 1

    # Array to store the 2D array of the solution
    u = np.zeros((n_t+2,n_x+2))

    # Fill in the element of the array
    # t = 0
    u[0] = u_t0.reshape(n_x+2)
    # Boundary
    u[1:,0] = u_bndry[:u_bndry.shape[0]//2].reshape(-1)
    u[1:,-1] = u_bndry[u_bndry.shape[0]//2:].reshape(-1)
    # Interior
    u[1:,1:-1] = u_interior.reshape((n_t+1,n_x))

    # Return
    return u


def visualize_predictions(network, f_u_exact, n_t, n_x, T=1,
        plot_types=['exact', 'pred', 'res'], **kwargs
    ):
    """
    Function to generate a heatmap visualizing the NNs predictions
    on the mesh
    """
    # Create the mesh for visualization
    mesh_interior, mesh_bndry, mesh_t0 = generate_mesh(n_t, n_x, T=T)

    # Compute the NN's prediction
    with torch.no_grad():
        u_pred_bndry = network(mesh_bndry)
        u_pred_t0 = network(mesh_t0)
    if 'pde_res' in plot_types:
        mesh_interior.requires_grad = True
        u_pred_interior = network(mesh_interior)
    else:
        with torch.no_grad():
            u_pred_interior = network(mesh_interior)

    # Compute the PDE residuals
    if 'pde_res' in plot_types:
        f_pde = kwargs['f_pde']
        pde_res_vec = f_pde(mesh_interior, u_pred_interior).detach()
        # Reshape
        pde_res = np.zeros((n_t+2,n_x+2))
        pde_res[1:,1:-1] = pde_res_vec.reshape(n_t+1,-1)
        # Detach the tensors - graph no longer needed
        mesh_interior = mesh_interior.detach()
        u_pred_interior = u_pred_interior.detach()

    # Reconstruct the NN's solution from the three pieces
    u_pred = reconstruct_u_array(
        u_pred_interior.detach(), u_pred_bndry, u_pred_t0
    )

    # Compute the exact solution
    u_exact_interior = f_u_exact(mesh_interior.detach())
    u_exact_bndry = f_u_exact(mesh_bndry)
    u_exact_t0 = f_u_exact(mesh_t0)
    u_exact = reconstruct_u_array(u_exact_interior, u_exact_bndry, u_exact_t0)

    # Heatmap visualization
    fig, axs  = plt.subplots(1,3,figsize=(15,5))
    for ax in axs:
        ax.set_xlabel('t')
        ax.set_ylabel('x')
    if 'suptitle' in kwargs:
        plt.suptitle(kwargs['suptitle'])

    vmxx_exact = np.abs(u_exact).max()
    vmxx_pred = np.abs(u_pred_interior.detach()).max()
    vmxx = max(vmxx_pred, vmxx_exact)

    # Subplots
    ims = []
    for plot_type, ax in zip(plot_types, axs):
        # Exact solution
        if plot_type == 'exact':
            ax.set_title('True solution')
            im = ax.imshow(
                u_exact.T, cmap='bwr_r', extent=(0,1,0,1),
                vmin=-vmxx, vmax=vmxx, origin='lower'
            )
        # Predicted solution
        elif plot_type == 'pred':
            ax.set_title('Predicted solution')
            im = ax.imshow(
                u_pred.T, cmap='bwr_r', extent=(0,1,0,1),
                vmin=-vmxx, vmax=vmxx, origin='lower'
            )
        # Residuals
        elif plot_type == 'res':
            residuals = u_exact - u_pred
            res_vmxx = np.abs(residuals).max()
            ax.set_title('Abs Val. Residuals')
            im = ax.imshow(
                np.abs(residuals).T, cmap='binary', extent=(0,1,0,1),
                origin='lower'
                # vmin=-res_vmxx, vmax=res_vmxx
            )
        # PDE Residuals
        elif plot_type == 'pde_res':
            res_vmxx = np.abs(pde_res).max()
            ax.set_title('Abs Val. PDE Residuals')
            im = ax.imshow(
                np.abs(pde_res).T, cmap='binary', extent=(0,1,0,1),
                origin='lower'
                # vmin=-res_vmxx, vmax=res_vmxx
            )

        # Save the heatmap
        ims.append(im)

    # Colorbors
    for i, im in enumerate(ims):
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.tight_layout()

    # Return the figure and axes
    return fig, axs
