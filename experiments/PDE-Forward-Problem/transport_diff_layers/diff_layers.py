# Set matplotlib for inline plots
# %matplotlib inline
# Set matplotlib default font size\n",
import matplotlib as mpl
mpl.rc('font', size=15)
# Imports
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

"""
Fully connected feed-forward neural network 

Network takes two arguments (x, t)
Network's output will be L-periodic in x,
    where L is an argument of the NN
Currently, x is a scalar

NOTE: I ended up not using this class
      It was easier just to impose periodic
      boundary conditions as part of the loss
"""

class SpacePeriodicNN(nn.Module):
    
    def __init__(self, N_h, L=1, activation='tanh'):
        super().__init__()
        
        # Save network parameters
        self.N_in = 2 # 1 for time, 1 for cosine, 1 for sine
        self.N_out = 1
        self.L = L
        
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
        
        
    def forward(self, s):
        # Extract x, t
        t = s[:,0]
        x = s[:,1]
        # Convert x to cos(2 pi x / L) and sine(2 pi x / L) 
        cos = torch.cos(2 * torch.pi * x / self.L)
        sin = torch.sin(2 * torch.pi * x / self.L)
        # Concatenate the input into a single tensor
        nn_input = torch.vstack((t,x,cos,sin)).t()
        # Run the input through the network
        nn_out = self.fcs(nn_input)
        nn_out = self.fch(nn_out)
        nn_out = self.fce(nn_out)
        return nn_out

"""
Fully connected feed-forward neural network 
"""

class FCNN(nn.Module):
    
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


"""
Functions for computing derivatives of the network's output wrt its input
"""

def compute_u_v(v, u):
    """
    Computes u_v for the independent variable v and dependent variable u
    """
    return torch.autograd.grad(u, v, torch.ones_like(u), create_graph=True)[0]


def compute_u_vv(v, u_v):
    """
    Computes u_vv for the independent variable 'v' and dependent variable 'u' (not a parameter here)
    Must be given u_v, the partial derivative computed using compute_u_v()
    """
    return torch.autograd.grad(u_v, v, torch.ones_like(u_v), create_graph=True)[0]


"""
Function to generate mesh points on [0,1]^d x [0,T]
"""
def generate_mesh(nt, nx, d=1, T=1):
    """
     ARGUMENTS
        
        nt, nx   :   Ints. Number of temporal and spatial interior points
                     for the mesh, respectively
                     
        d        :   Int. Spatial dimension
        
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
        [t_vals] + [x_vals] * d#, indexing='ij'
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

def reconstruct_u_array(u_interior, u_bndry, u_t0):
    """
    Function to reconstruct the (n_t+2, n_x+2) solution
    given the solution at t=0, on the boundaries, and 
    inside the mesh
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
                vmin=-vmxx, vmax=vmxx
            )
        # Predicted solution
        elif plot_type == 'pred':
            ax.set_title('Predicted solution')
            im = ax.imshow(
                u_pred.T, cmap='bwr_r', extent=(0,1,0,1),
                vmin=-vmxx, vmax=vmxx
            )
        # Residuals
        elif plot_type == 'res':
            residuals = u_exact - u_pred
            res_vmxx = np.abs(residuals).max()
            ax.set_title('Abs Val. Residuals')
            im = ax.imshow(
                np.abs(residuals).T, cmap='binary', extent=(0,1,0,1),
                # vmin=-res_vmxx, vmax=res_vmxx
            )
        # PDE Residuals
        elif plot_type == 'pde_res':
            res_vmxx = np.abs(pde_res).max()
            ax.set_title('Abs Val. PDE Residuals')
            im = ax.imshow(
                np.abs(pde_res).T, cmap='binary', extent=(0,1,0,1),
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


"""
Mesh-free training algorithm, specific to the SpacePeriodicNN class
"""
def train_on_mesh(network, f_pde, f_ic, nt, nx, 
        alpha=None, n_iters=1000, lr=1e-3, verbose=None, u_exact=None,
        make_pred_figs=None, fig_dir='./figures', fig_fname='transport',
        fig_mesh_size=None
    ):
    """
    ARGUMENTS
    
        network      :  FCN network to be trained
        
        f_pde        :  Callable. Function that computes the partial derivative based 
                        quantity that should be minimized
        
        f_ic         :  Callable. Function that computes the initial condition 
                        for the PDE that the NN is being trained to solve
                        
        nt, nx       :  Ints. Number of temporal and spatial interior points
                        for the mesh, respectively
                        
        alpha        :  Tuple. Sets the scaling of the three
                        terms in the loss function
                        
                        L = alpha[0] L_pde + alpha[1] * L_ic

                        Default: [1,1]
        
        n_iters      :  Number of training iterations
                        Default: 1000
                        
        lr           :  Learning rate
                        Default: 1e-3
        
        u_exact      :  None or Callable function that computes the exact
                        solution given any point in the domain
                        Default: None
                        
        make_pred_figs      :   None or Int.
                                Set to Int to set the rate at which to 
                                    generate and save figures visualizing 
                                    the NNs predictions and errors. 
                                    Requires 'u_exact' to not be None
                                Set to None to not create such figures.
                                Default: None
                                    
        fig_dir      :  String. Path to directory where training visualizations
                        should be saved.
                        Default: './figures'
        
        fig_fname    :  String. Figures generated by this method will have
                        the filenames of the form 'save_fig_fname_iter%d.png'
                        
        fig_mesh_size   : Tuple (n_t_viz, n_x_viz) setting the mesh size used when
                          generating the visualizations of the networks predictions.
                          Set to None to use (n_t, n_x)
                          Default: None
                            
    RETURNS
    
        L_pde    
        
        L_bc
        
        L_ic
        
        L_train
        
        L2_err
    """
    
    # Loss coefs alpha[i]
    if alpha is None:
        alpha = [1,1]
    alpha = np.array(alpha, dtype=float)
    
    # Create the optimizer
    optimizer = torch.optim.Adam(network.parameters(),lr=lr)
    
    # Object to compute MSE loss
    mse_loss = nn.MSELoss()
    
    # Arrays to store the losses
    L_pde = np.zeros(n_iters)    # PDE loss
    L_bc = np.zeros(n_iters)     # Initial condition loss
    L_ic = np.zeros(n_iters)     # Initial condition loss
    L_train = np.zeros(n_iters)  # Overall training loss
    if u_exact is not None:
        L2_err = np.zeros(n_iters)  # L2-error on exact solution
        
    # Create the mesh
    mesh_interior, mesh_boundary, mesh_t0 = generate_mesh(nt, nx)
    # Set the interior points to requires grad
    mesh_interior.requires_grad = True
    # Compute the number of boundary points
    n_bndry_pts = mesh_boundary.shape[0]
    # Compute the initial condition at the mesh points
    u_exact_t0 = f_ic(mesh_t0)
    # Compute the exact solution at the mesh points
    # if the function for computed the exact soln is
    # provided
    if u_exact is not None:
        u_exact_interior = u_exact(mesh_interior.detach())
        
    # Mesh figsize
    if make_pred_figs is not None:
        if fig_mesh_size is None:
            n_t_viz, n_x_viz = (nt, nx)
        else:
            n_t_viz, n_x_viz = fig_mesh_size
    
    # Training loop
    for i in range(n_iters):
        # Clear the parameter gradients
        optimizer.zero_grad()
        
        # Initial condition loss
        # Compute the network's predictons for mesh_t0
        u_net_t0 = network(mesh_t0)
        # Initial condition loss
        L_ic_i = mse_loss(u_exact_t0, u_net_t0) 
        L_ic[i] = L_ic_i.detach() 
        
        # Periodic BC Loss
        u_net_bndry = network(mesh_boundary)
        u_net_left_bndry = u_net_bndry[:n_bndry_pts//2]
        u_net_right_bndry = u_net_bndry[n_bndry_pts//2:]
        L_bc_i = mse_loss(u_net_left_bndry, u_net_right_bndry)
        L_bc[i] = L_bc_i.detach() 
        
        # PDE loss
        # Compute the NN's output at the interior points
        u_net_interior = network(mesh_interior)
        # Compute the NN's PDE loss
        L_pde_i = f_pde(mesh_interior, u_net_interior).square().mean()
        L_pde[i] = L_pde_i.detach()
        
        # L2 Error on exact solution
        if u_exact is not None:
            # Compute the network's L2 error on these points
            L2_err[i] = np.sqrt(
                mse_loss(u_exact_interior, u_net_interior.detach())
            )
        
        # Overall loss
        L_train_i = alpha[0] * L_pde_i + alpha[1] * L_bc_i + alpha[2] * L_ic_i
        L_train[i] = L_train_i.detach()
        
        # Update the network's parameter
        L_train_i.backward()
        optimizer.step()
        
        # Progress update
        if verbose and ((i == 0) or ((i+1) % verbose) == 0):
            fmt_str = 'Iter %d/%d, Losses: PDE=%.2e  BC=%.2e  IC=%.2e  TR=%.2e'
            msg = fmt_str % (i+1, n_iters, L_pde_i, L_bc_i, L_ic_i, L_train_i)
            if u_exact is not None:
                msg += ' L2-err: %.2e' % L2_err[i]
            # print(msg)
            
        # Save visualization of network predictions
        if make_pred_figs and ((i == 0) or ((i+1) % make_pred_figs) == 0):
            # Make the figure
            fig, axs = visualize_predictions(
                network, u_exact, n_t_viz, n_x_viz,
                suptitle='Training Iteration %d' % (i+1),
                f_pde=f_pde, plot_types=['pred', 'pde_res', 'res']
            )
            # Save to file
            plt.savefig(
                '%s/%s_iter%d.png' % (fig_dir, fig_fname, i+1),
                facecolor='white', transparent=False
            )
            # Close the figure
            plt.close()
            
            
    # Return the loss arrays as np arrays
    losses = [L_pde, L_bc, L_ic, L_train]
    if u_exact is not None:
        losses.append(L2_err)
    return losses

def f_pde_transport(v, u, a=1):
    """
    PDE loss associated with the transport equation
    u_t + a u_x = 0
    
    Only works for d = 1 right now
    
    ARGUMENTS
    
        v    :   (# points, d+1)-Torch tensor of (t[i], x[i]) pairs where
                     v[i,0] = t[i] and s[i,1:] = x[i]
                     
        u    :   (# points,)-Torch tensor of u[i] = u(t[i], x[i])
        
        a    :   (d,)-Torch tensor. The transport velocity
                 Default: a = [1, 1, ..., d]^t
        
    RETURNS
    
       the PDE loss ... IDK what to call it
    """
    # Set the Transport velocity
    if a is None:
        a = torch.ones(v.shape[1]-1)
    
    # Compute u_t
    u_v = compute_u_v(v, u)
    # Extract u_t and u_x
    u_t = u_v[:,0]
    u_x = u_v[:,1]
    
    # Compute Q
    return u_t + a * u_x



def f_ic_basic_transport(s):
    """
    Basic IC for the transport equation
    Only works for d = 1
    
    s = (t, x)
    """
    return torch.cos(2 * np.pi * s[:,1]).view(-1,1)


def u_exact_basic_transport(s):
    """
    Exact solution to the transport equation
    
                u_t + u_x = 0
    
    on [0,1] x [0,1] with the basic initial 
    condition
    
            u(x,0) = cos(2 pi x)
    """
    x_t = s[:,1] - s[:,0]
    return torch.cos(2 * np.pi * x_t).view(-1,1)

"""
------------------------------------------------------------------------
                                The TEST
------------------------------------------------------------------------
"""

"""
Parameters
"""
# list of number of layers for network
layer_list = [1,5,10]
# list of number of neurons in each layer
neuron_list = [2,50,100]
# list of seeds 
seed_list = [1,10,100,1000,10000,100000,200000,300000,400000,1000000]

'''
Iterating through each number of layers, the number of neurons in each layer and the seed.
'''
# Training parameters
nt = 50
nx = 20
alpha = np.array([1,1.5,1])       # Relative importance of pde / bc / ic loss
n_iters = 900     # Number of training iterations
lr = 1e-3          # Learning rate
verbose=30

N_h = []
for layers in tqdm(layer_list):
    for neurons in tqdm(neuron_list):
        for l in range(int(layers)):
            N_h.append(int(neurons))
        for seed in tqdm(seed_list):
            # Network parameters
            # Visualization parameters (during training)
            make_pred_figs=15
            # fig_dir='/Users/dvirblander/Desktop/Fall\ 2021/Research\ Project\ Fall\ Semester/Umass-Math691Y-F21-RNNs/experiments/PDE-Forward-Problem/'
            # fig_dir = fig_dir.replace('\\', '')
            fig_dir_b = os.getcwd()
            # print(fig_dir_b)
            # /Users/dvirblander/Desktop/Fall 2021/Research Project Fall Semester/UMass-Math691Y-F21-RNNs/experiments/PDE-Forward-Problem/transport_diff_layers/figures/transport
            fig_dir = '/Users/dvirblander/Desktop/Fall 2021/Research Project Fall Semester/UMass-Math691Y-F21-RNNs/experiments/PDE-Forward-Problem/transport_diff_layers/figures/transport'
            fig_fname='transport for '+str(layers) + 'layers, ' + str(neurons) + ' per layer, and ' + str(seed) + 'seed'
            fig_mesh_size=(50,50)
            # Set the random seed for reproducibility
            seed = int(seed)
            # print('Random seed:', seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Create the network
            network = FCNN(N_h=N_h)

            # Clear the figure directory
            for f in os.listdir(fig_dir):
                if f.startswith(fig_fname):
                    os.remove(os.path.join(fig_dir, f))

            """
            Training Loop
            """
            L_pde, L_bc, L_ic, L_train, L2_err = train_on_mesh(network,
                    lambda pts,u: f_pde_transport(pts, u), f_ic_basic_transport, 
                    nt, nx, alpha, n_iters, lr, verbose, u_exact_basic_transport,
                    make_pred_figs, fig_dir, fig_fname, fig_mesh_size
            )
            """
            Plot the learning curves
            """
            lc_title = 'Learning curves (LCs)'+fig_fname
            plt.figure(figsize=(10,6))
            plt.title(lc_title)
            plt.xlabel('Training time (# parameter updates)')
            plt.ylabel('log LC')

            LCs = [L_pde, L_bc, L_ic, L_train, L2_err]
            labels = ['PDE', 'BC', 'IC', 'Train', 'L2 Error']
            sample_rate = 1
            times = sample_rate * np.arange(LCs[0].shape[0] / sample_rate)
            for lc,label in zip(LCs, labels):
                plt.scatter(times, np.log(lc[::sample_rate]), label=label, s=5)
            plt.legend()
            plt.savefig(lc_title+'.png')

            """
            Post training visualization
            """
            fig, axs = visualize_predictions(
                network, u_exact_basic_transport, n_t=50, n_x=50, T=1,
                suptitle='Post Training Predictions and Residuals'
            )
            suptitle = 'Post Training Predictions and Residuals for ' + fig_fname
            plt.savefig(suptitle+'.png')



# # Network parameters
# N_h = [128,128]

# # Training parameters
# nt = 50
# nx = 20
# alpha = np.array([1,1.5,1])       # Relative importance of pde / bc / ic loss
# n_iters = 900     # Number of training iterations
# lr = 1e-3          # Learning rate
# verbose=30

# # Visualization parameters (during training)
# make_pred_figs=15
# fig_dir='./figures/transport'
# fig_fname='transport'
# fig_mesh_size=(50,50)

# """
# Setup
# """
# # Set the random seed for reproducibility
# seed = np.random.randint(1e6) # 913988
# print('Random seed:', seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# # Create the network
# network = FCNN(N_h=N_h)

# # Clear the figure directory
# for f in os.listdir(fig_dir):
#     if f.startswith(fig_fname):
#         os.remove(os.path.join(fig_dir, f))

# """
# Training Loop
# """
# L_pde, L_bc, L_ic, L_train, L2_err = train_on_mesh(network,
#         lambda pts,u: f_pde_transport(pts, u), f_ic_basic_transport, 
#         nt, nx, alpha, n_iters, lr, verbose, u_exact_basic_transport,
#         make_pred_figs, fig_dir, fig_fname, fig_mesh_size
#     )