# Change root directory
import sys, os
sys.path.append(os.path.abspath('../../'))

# Package Imports
from time import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA

# Local Imports
from src.baseline_models import RNN
from src.load_data import ERT_germany
from src.train_and_eval import train
from src.clustering import hierarchical_clustering

"""
--------------------------------------------------------------------------------
    Parameters
--------------------------------------------------------------------------------
"""
# Dataset parameters
dtype = 'cases_daily'
T = 100
country_wide_avg = True

# Model parameters
N_h = 32
p_hh = 1
g_hh = 1.
f_h = 'tanh'
f_out = 'tanh'
sigma = 0e-3 # noise parameter
tau = 1
learn_tau = False

# Training parameters
n_epochs = 3 * 256
lr = 1e-3
lr_decay = 0.5
weight_decay = 0.0
batches_til_first_lr_update=256
clip_gradient_thres=None
verbose = True

"""
--------------------------------------------------------------------------------
    Setup
--------------------------------------------------------------------------------
"""
# Random Seed
seed = np.random.randint(1e6) # 685138
print('Random seed: %d' % seed)
np.random.seed(seed)

# Load the dataset
# directory where the data is stored, relative to this script
datadir = '../../euro_regional_tracker/germany_data/data'
# load the data from file
y = ERT_germany(dtype=dtype, seq_len=T, country_wide_avg=country_wide_avg,
    datadir=datadir
)
# Save the sequence length
T = y.shape[0]
# Create the input sequence (all zeros for now)
u = torch.zeros(1, T, 0)
# Reshape y
y = y.view(1, T, 1)
# Normalize to -1,1
y -= y.min()
y /= 0.5 * y.max()
y -= 1

# Create the RNN model
network = RNN(
    0, N_h, 1, # Only one output for now
    f_h=f_h, f_out=f_out, g_hh=g_hh, p_hh=p_hh,
    tau=tau, learn_tau=learn_tau
)

"""
--------------------------------------------------------------------------------
    Training
--------------------------------------------------------------------------------
"""
losses, = train(network, u, y, n_epochs, 1, lr, lr_decay=lr_decay,
        weight_decay=weight_decay,
        batches_til_first_lr_update=batches_til_first_lr_update,
        clip_gradient_thres=clip_gradient_thres, verbose=True
)

# Plot the learning curve
plt.figure(figsize=(6,4))
plt.title('Learning Curve')
plt.xlabel('Training Time (# epochs)')
plt.ylabel('MSE')
plt.plot(losses)


"""
--------------------------------------------------------------------------------
    Visualization
--------------------------------------------------------------------------------
"""
# Visualize predictions until user tells us otherwise
T_viz = T

# Make a prediction
with torch.no_grad():
    hd, y_pred = network(torch.zeros(1,T_viz,0), return_dynamics=True)

# Reshape and normalize
H = hd.detach().numpy().reshape(-1, N_h)
# H = (H - np.mean(H, axis=0)) # / np.std(H, axis=0)

# Target task for T=T_viz
y_viz = y

"""
Visualize predictions
"""
plt.figure(figsize=(8,3))
plt.title('Example Network Predictions')
plt.xlabel('Time t (# timesteps)')
plt.ylabel('Memory Amplitude')
plt.xlim(0,T_viz-1)
Dy, dy = (1, 0.1)
n_mem = 1
for i in range(n_mem):
    # Plot output target for memory i
    plt.fill_between(
        np.arange(T_viz), y_viz[0,:,i] + (Dy+dy)*i+dy, (Dy+dy)*i+dy,
        step='mid', alpha=0.5
    )
    # Plot network prediction for memory i
    plt.plot(
        np.arange(T_viz), y_pred[0,:,i] + (Dy+dy)*i+dy,
        ds='steps-mid', c='k', linewidth=2
    )

# Reshape and normalize
H = hd.detach().numpy().reshape(-1, N_h)
H_centered = (H - np.mean(H, axis=0)) # / np.std(H, axis=0)


"""
Dynamics heatmap
"""

# Clustering
order = hierarchical_clustering(H_centered)

# Reorder dynamics heatmap
h_viz = np.zeros_like(H.T)
# Reorder columns of h
for j in range(N_h):
    h_viz[j] = H[:,order[j]]

plt.figure(figsize=(12,6))
plt.title('Hidden layer dynamics')
plt.xlabel('Time t (# timesteps)')
plt.ylabel('hidden unit #')
vmxx = np.max(np.abs(h_viz))
im = plt.imshow(h_viz, aspect='auto', interpolation='none', cmap='bwr_r',
    vmin=-1,vmax=1
)
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="2%", pad=0.05)
plt.colorbar(im, cax=cax)


"""
Recurrent weights heatmap
"""
to_copy = network.W_hh.detach().numpy().T
W_hh = np.zeros((N_h, N_h))
for i in range(N_h):
    for j in range(N_h):
        W_hh[i,j] = to_copy[order[i], order[j]]
plt.figure(figsize=(6,6))
plt.title('Hidden-to-Hidden Weights')
vmxx = np.max(np.abs(W_hh))
im = plt.imshow(
    W_hh, interpolation='none', cmap='bwr_r', vmin=-vmxx, vmax=vmxx
)
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="2%", pad=0.05)
plt.colorbar(im, cax=cax)


"""
PCA of Dynamics
"""

fig, ax = plt.subplots(1,1,figsize=(6,6))
fig.suptitle('Low-dimensional visualization of dynamics')

# Subplot 1: PCA
pca = PCA(n_components=5)
H_pca = pca.fit_transform(h_viz.T)

ax0 = fig.add_subplot(111, projection='3d')
ax0.set_title('PCA of network dynamics')
ax0.set_xlabel('PCA 0')
ax0.set_ylabel('PCA 1')
ax0.set_zlabel('PCA 2')
PCA_inds = [0,1,2]
ax0.scatter(
    H_pca[:, PCA_inds[0]],
    H_pca[:, PCA_inds[1]],
    H_pca[:, PCA_inds[2]]
)


# # Subplot 2: Projection onto top three eigenvectors
# ax1 = fig.add_subplot(111, projection='3d')
# ax1.set_title('PCA of network dynamics')
# ax1.set_xlabel('PCA 0')
# ax1.set_ylabel('PCA 1')
# ax1.set_zlabel('PCA 2')
# PCA_inds = [0,1,2]
# sc_ToD = ax0.scatter(
#     H_pca[:, PCA_inds[0]],
#     H_pca[:, PCA_inds[1]],
#     H_pca[:, PCA_inds[2]],
# )



# Show the plots
plt.show()
