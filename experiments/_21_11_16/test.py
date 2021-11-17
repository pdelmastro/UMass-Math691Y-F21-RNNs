# Change root directory
import sys, os
sys.path.append(os.path.abspath('../../'))

# Package Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA

# MLP text size
import matplotlib
matplotlib.rc('font', size=16)

# Local Imports
from src.baseline_models import RNN
from src.load_data import ERT_germany
from src.train_and_eval import train
from src.clustering import hierarchical_clustering

"""
---------------------------------------------------------------------
    Parameters
---------------------------------------------------------------------
"""

# Dataset parameters
dtype = 'cases_daily'
T = 100
tau_minus = 5 # Sliding window size for inputs
tau_plus  = 1 # Slide window for outputs. Currently un-used
country_wide_total = False
per_10k_population = False
normalize = False
squash_to = (-1,1)
smooth = 7
p_tr = 0.8    # Portion (0-1) of the dataset used for training

# Model parameters
N_h = 64
p_hh = 1
g_hh = 1
f_h = 'tanh'
f_out = 'tanh'
sigma = 0e-3 # noise parameter
tau = 2
learn_tau = False

# Training parameters
n_epochs = 3 * 256
batch_size = 16
lr = 5e-4
lr_decay = 0.75
weight_decay = 0.0
batches_til_first_lr_update=256
clip_gradient_thres=None
verbose = True

"""
---------------------------------------------------------------------
    Setup
---------------------------------------------------------------------
"""

# Function to compute a sliding windows of the input time-series
def compute_SW(y, tau):
    """
    Given a tensor y of shape (N,T), returns a new tensor SW
    of show (N,T,tau) where
    
                SW[i,t,k] =   0          if t-k < 0
                              y[i,t-k]   else
    
    where k = 0, 1, ..., tau-1
    """
    N,T = y.shape
    SW = -torch.ones((N,T,tau))
    for k in range(tau):
        SW[:,k:,k] = y[:,:(T-k)]
    return SW


# Random Seed
seed = np.random.randint(1e6) # 349217
print('Random seed: %d' % seed)
np.random.seed(seed)

# Load the dataset
# directory where the data is stored, relative to this script
datadir = '../../euro_regional_tracker/germany_data/data'
# load the data from file
y_all = ERT_germany(dtype=dtype, seq_len=T, 
    smooth=smooth, squash_to=squash_to,
    per_10k_population=per_10k_population,
    country_wide_total=country_wide_total,
    normalize=normalize, datadir=datadir
)

# Create the target output sequences
#  (Same as y_all, but starting at day tau_plus
#   so the network learns to make predictions for
#   the future)
y = y_all.view(-1,T,1)[:,tau_plus:].contiguous()

# Create the input sequence
# Note: We need to truncate the sequence to length 
#   T-tau_plus to match the length of the target output
#   sequences
u = compute_SW(y_all, tau_minus)[:,tau_plus:]

# Create the RNN model
network = RNN(
    tau_minus, N_h, 1,
    f_h=f_h, f_out=f_out, g_hh=g_hh, p_hh=p_hh,
    tau=tau, learn_tau=learn_tau
)

"""
---------------------------------------------------------------------
    Training
---------------------------------------------------------------------
"""

# Split into a train and test sets
n_train = int(p_tr * y.shape[0])
u_train, u_test = u[:n_train], u[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

print('Train set: %d examples' % n_train)
print('Test  set: %d examples' % (u.shape[0]-n_train))


losses, = train(network, u_train, y_train, n_epochs, batch_size, lr, 
        lr_decay=lr_decay, weight_decay=weight_decay,
        batches_til_first_lr_update=batches_til_first_lr_update,
        clip_gradient_thres=clip_gradient_thres, verbose=True
)