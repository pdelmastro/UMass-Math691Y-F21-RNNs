"""
Functions for training PINNs
"""
# Library imports
import numpy as np
import torch
import torch.nn

# Local imports
from .mesh import generate_mesh


"""
--------------------------------------------------------------------------------
    Generic PINN Optimizer
--------------------------------------------------------------------------------
"""
class PINN_Optimizer(object):

    """
    ----------------------------------------------------------------------------
        Setup functions
    ----------------------------------------------------------------------------
    """
    def __init__(self):
        pass

    def setup(self, *args, **kwargs):
        """
        Sets up various data structures used by the optimizer during the
        training loop (self.train())
        """
        pass


    """
    ----------------------------------------------------------------------------
        Domain Sampling Functions
    ----------------------------------------------------------------------------
    """
    def next_t0_condition(self, *args, **kwargs):
        """
        Returns a tuple of data used for evaluating the initial condition
        residuals on the next iteration of training

        ARGUMENTS

            None required by this class. The error based sampling optimizer
            is the only class that needs arguments for this function.

        RETURNS

            t0_points       :   (# points, self.d+1)-Torch tensor. Points at
                                which the initial condition is to be evaluated.
                                These are pairs t0_points[i] = [t[i]; x[i]]

            u_exact_t0      :   (# points, 1)-Torch tensor. Initial condition at
                                t=t[i], x=x[i]
        """
        pass


    def next_boundary_condition(self, *args, **kwargs):
        """
        Returns a tuple of data used for evaluating the boundary condition
        residuals on the next iteration of training.

        ARGUMENTS

            None required by this class. The error based sampling optimizer
            is the only class that needs arguments for this function.

        RETURNS

            t0_points       :   (# points, self.d+1)-Torch tensor. Points at
                                which the initial condition is to be evaluated.
                                These are pairs t0_points[i] = [t[i]; x[i]]

            u_exact_t0      :   (# points, 1)-Torch tensor. Initial condition at
                                t=t[i], x=x[i]
        """
        pass


    def next_boundary_pairs(self, *args, **kwargs):
        """
        Returns the boundary points of the domain at which the BC residuals
        will be evaluated on the next iteration of training. This function is
        used when imposing periodic boundary conditions.

        ARGUMENTS

            None required by this class. The error based sampling optimizer
            is the only class that needs arguments for this function.

        RETURNS

            bndry_pts   :       (P, self.d+1)-Torch tensor. Points at which the
                                  periodic BC residuals will be evaluated.
                                  These are pairs
                                        bndry_pts[i] = [t[i]; x_bndry[i]]

            opp_pts     :       (P, self.d+1)-Torch tensor. Points at which are
                                  on the boundary opposite from 'bndry_pts'
                                  These are pairs
                                        opp_pts[i] = [t[i]; x_opp[i]],
                                  where position x_opp[i] is on the opposite
                                  boundary from position x_bndry[i].
                                For instance, in the 1d case, if
                                    x_bndry[i] = 0 then x_opp[i] = 1.
        """
        pass


    def next_interior_points(self, *args, **kwargs):
        """
        Returns the interior points of the domain at which the PDE residuals
        will be evaluated on the next iteration of training

        ARGUMENTS

            None required by this class. The error based sampling optimizer
            is the only class that needs arguments for this function.

        RETURNS

            interior_pts   :   (# points, self.d+1)-Torch tensor. Points at
                                which the PDE residuals will be evaluated.
                                These are pairs interior_pts[i] = [t[i]; x[i]]
        """
        pass



    """
    ----------------------------------------------------------------------------
        Training Functions
    ----------------------------------------------------------------------------
    """

    def step(*args, **kwargs):
        """
        TODO: explain - this is the function called at the end of the iteration
            used by the error based sampling method to update its internal
            state

        Does nothing for the mesh based optimizer, which is the default defined
        here in PINN_Optimizer
        """
        pass


    def train(self,
            network, f_pde, f_ic, f_bc=None, alpha=None, n_iters=1000, lr=1e-3,
            f_u_exact=None, eval_mesh_size=(20,20), verbose=None,
            make_pred_figs=None, fig_dir='./figures', fig_fname='transport',
            fig_mesh_size=(20,20)
        ):
        """
        ARGUMENTS

            network      :  Network to be trained

            f_pde        :  Callable. Function that computes the partial
                            derivative based quantity that should be minimized

            f_ic         :  Callable. Function that computes the initial
                              condition for the problem that the NN is being
                              trained to solve

            f_bc         :  Callable or None.
                            If Callable, Function that computes the boundary
                              condition for the problem that the NN is being
                              trained to solve
                            If None, this method will impose periodic boundary
                              conditions
                            Default: None

            alpha        :  Tuple. Sets the scaling of the three
                            terms in the loss function

                              L = alpha[0] L_pde + alpha[1] L_bc + alpha[2] L_ic

                            Default: [1,1,1]

            n_iters      :  Number of training iterations
                            Default: 1000

            lr           :  Learning rate
                            Default: 1e-3

            f_u_exact    :  None or Callable function that computes the exact
                            solution given any point in the domain
                            Default: None

            eval_mesh_size  :   Tuple (n_t_ev, n_x_ev) setting the mesh size
                                used evaluatng the network's L2-error
                                Default: (20, 20)

            verbose         :   Bool. Set to True to print out progress updates
                                while training.

            make_pred_figs      :   None or Int.
                                    Set to Int to set the rate at which to
                                        generate and save figures visualizing
                                        the NNs predictions and errors.
                                        Requires 'u_exact' to not be None
                                    Set to None to not create such figures.
                                    Default: None

            fig_dir      :  String. Path to directory where training
                            visualizations should be saved.
                            Default: './figures'

            fig_fname    :  String. Figures generated by this method will have
                            the filenames of the form
                            'save_fig_fname_iter%d.png'

            fig_mesh_size   :   Tuple (n_t_viz, n_x_viz) setting the mesh size
                                used when generating the visualizations of the
                                networks predictions.
                                Default: (20, 20)

        RETURNS

            L_pde       :   (# iterations,)-np array. PDE loss of the model
                            at the end of every iteration of training, evaluated
                            at the training points

            L_bc        :   (# iterations,)-np array. BC loss of the model
                            at the end of every iteration of training, evaluated
                            at the training points

            L_ic        :   (# iterations,)-np array. IC loss of the model
                            at the end of every iteration of training, evaluated
                            at the training points

            L_train     :   (# iterations,)-np array. Overall loss of the model
                            at the end of every iteration of training, evaluated
                            at the training points

            L2_err      :   (# iterations,)-np array. L2 error of the model
                            at the end of every iteration of training, evaluated
                            on the testing mesh.
                            Only returned when 'f_u_exact' is not None

        TODO
        - Add (cosine) learning rate scheduler
        - Add option to set initial condition for u_t (needed for wave eqn)
        - Option to control time T and dimension d
        """
        # Loss coefs alpha[i]
        if alpha is None:
            alpha = [1,1,1]
        alpha = np.array(alpha, dtype=float)

        # Create the optimizer
        optimizer = torch.optim.Adam(network.parameters(),lr=lr)

        # Arrays to store the losses
        L_pde = np.zeros(n_iters)    # PDE loss
        L_bc = np.zeros(n_iters)     # Initial condition loss
        L_ic = np.zeros(n_iters)     # Initial condition loss
        L_train = np.zeros(n_iters)  # Overall training loss
        if f_u_exact is not None:
            L2_err = np.zeros(n_iters)  # L2-error on exact solution
            # Generate the evaluation mesh
            nt_ev,nx_ev = eval_mesh_size
            eval_mesh = generate_mesh(nt_ev, nx_ev)
            # Exact solution on the mesh
            u_exact_eval = [f_u_exact(pts) for pts in eval_mesh]

        # Perform any setup specific to the optimizer
        self.setup(f_ic, f_bc)

        # Training loop
        for i in range(n_iters):
            # Clear the parameter gradients
            optimizer.zero_grad()

            # Initial condition loss
            # Get the set of points at which the initial condition loss
            # will be evaluated, as well as the exact solution at these points
            t0_pts, u_exact_t0 = self.next_t0_condition(f_ic)
            # Compute the network's predictons for these points
            u_net_t0 = network(t0_pts)
            # Initial condition loss
            res_ic_i = u_exact_t0 - u_net_t0
            L_ic_i = res_ic_i.square().mean()
            L_ic[i] = L_ic_i.detach()

            # Boundary condition loss
            # Case 1: Periodic BCs
            if f_bc is None:
                # Get pairs of points from the boundary
                bndry_pts, opp_bndry_pts = self.next_boundary_pairs()
                # Compute the network's output on these boundary points
                u_net_bndry = network(bndry_pts)
                u_net_opp_bndry = network(opp_bndry_pts)
                # Compute the residuals
                res_bc_i = u_net_bndry - u_net_opp_bndry
            # Case 2: Dirichlet BCs
            else:
                # Get the set of points at which the initial condition loss
                # will be evaluated, as well as the exact solution at these pts
                bndry_pts, u_exact_bndry = self.next_boundary_condition(f_bc)
                # Compute the network's predictons for these points
                u_net_bndry = network(bndry_pts)
                # Initial condition loss
                res_bc_i = u_exact_bndry - u_net_bndry
            # Compute the loss based on the residuals
            L_bc_i = res_bc_i.square().mean()
            L_bc[i] = L_bc_i.detach()

            # PDE loss
            # Get the set of points at which the PDE loss will be evaluated
            interior_pts = self.next_interior_points()
            # Compute the NN's output at the interior points
            u_net_interior = network(interior_pts)
            # Compute the NN's PDE loss
            res_pde_i = f_pde(interior_pts, u_net_interior)
            L_pde_i = res_pde_i.square().mean()
            L_pde[i] = L_pde_i.detach()

            # L2 Error on exact solution
            if f_u_exact is not None:
                # Compute the network's output on the evaluation mesh
                with torch.no_grad():
                    u_net_eval = [network(pts) for pts in eval_mesh]
                # Compute the L2-error
                L2s_eval_i = [
                    (u_net-u_exact).square().mean()
                    for u_net, u_exact in zip(u_net_eval, u_exact_eval)
                ]
                L2_err[i] = torch.sqrt(sum(L2s_eval_i))

            # Overall loss
            L_train_i = alpha[0] * L_pde_i + alpha[1] * L_bc_i + alpha[2] * L_ic_i
            L_train[i] = L_train_i.detach()

            # Update the network's parameter
            L_train_i.backward()
            optimizer.step()

            # Post-iteration updated based on the losses
            # This functionality is used by the error-based training algorithm
            self.step(res_ic_i.detach(), res_bc_i.detach(), res_pde_i.detach())

            # Progress update
            if verbose and ((i == 0) or ((i+1) % verbose) == 0):
                fmt_str = 'Iter %d/%d, Losses: PDE=%.2e  BC=%.2e  IC=%.2e  TR=%.2e'
                msg = fmt_str % (i+1, n_iters, L_pde_i, L_bc_i, L_ic_i, L_train_i)
                if f_u_exact is not None:
                    msg += ' L2-err: %.2e' % L2_err[i]
                print(msg)

            # Save visualization of network predictions
            if make_pred_figs and ((i == 0) or ((i+1) % make_pred_figs) == 0):
                # Make the figure
                fig, axs = visualize_predictions(
                    network, f_u_exact, n_t_viz, n_x_viz,
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
        if f_u_exact is not None:
            losses.append(L2_err)
        return losses


"""
--------------------------------------------------------------------------------
    Mesh Based Optimizer
--------------------------------------------------------------------------------
"""

class MeshBasedOptimizer(PINN_Optimizer):

    """
    ----------------------------------------------------------------------------
        Setup functions
    ----------------------------------------------------------------------------
    """
    def __init__(self, nt, nx, d=1, T=1):
        """
        ARGUMENTS

            nt, nx   :  Ints. Number of temporal and spatial interior points
                        for the mesh, respectively

            d        :  Int. Spatial dimension
                        Default: 1
                        NOTE: Only works for d=1 currently

            T        :  Float. Sets time domain to [0,T]
                        Default: 1
        """
        self.nt = nt
        self.nx = nx
        self.d = d
        self.T = T

        # Create the training mesh
        mesh = generate_mesh(nt, nx, d, T)
        self.mesh_interior, self.mesh_bndry, self.mesh_t0 = mesh
        # Set the interior points to requires grad
        self.mesh_interior.requires_grad = True

        # Variables storing the number of training points for each part
        # of the mesh
        self.n_pde_pts = self.mesh_interior.shape[0]
        self.n_bc_pts = self.mesh_bndry.shape[0]
        self.n_ic_pts = self.mesh_t0.shape[0]


    def setup(self, f_ic, f_bc):
        """
        Sets up various data structures used by the optimizer during the
        training loop (self.train())
        """
        self.f_ic = f_ic
        self.f_bc = f_bc

        # Compute the initial condition on the training mesh
        self.u_exact_t0 = f_ic(self.mesh_t0)

        # Compute the boundary condition on the training mesh if
        # the boundary conditions aren't periodic
        if self.f_bc is not None:
            self.u_exact_bndry = f_bc(self.mesh_bndry)


    """
    ----------------------------------------------------------------------------
        Domain Sampling Functions
    ----------------------------------------------------------------------------
    """
    def next_t0_condition(self, *args, **kwargs):
        """
        Returns a tuple of data used for evaluating the initial condition
        residuals on the next iteration of training

        ARGUMENTS

            None required by this class. The error based sampling optimizer
            is the only class that needs arguments for this function.

        RETURNS

            t0_points       :   (# points, self.d+1)-Torch tensor. Points at
                                which the initial condition is to be evaluated.
                                These are pairs t0_points[i] = [t[i]; x[i]]

            u_exact_t0      :   (# points, 1)-Torch tensor. Initial condition at
                                t=t[i], x=x[i]
        """
        return self.mesh_t0, self.u_exact_t0


    def next_boundary_condition(self, *args, **kwargs):
        """
        Returns a tuple of data used for evaluating the boundary condition
        residuals on the next iteration of training.

        ARGUMENTS

            None required by this class. The error based sampling optimizer
            is the only class that needs arguments for this function.

        RETURNS

            t0_points       :   (# points, self.d+1)-Torch tensor. Points at
                                which the initial condition is to be evaluated.
                                These are pairs t0_points[i] = [t[i]; x[i]]

            u_exact_t0      :   (# points, 1)-Torch tensor. Initial condition at
                                t=t[i], x=x[i]
        """
        return self.mesh_boundary, self.u_exact_bndry


    def next_boundary_pairs(self, *args, **kwargs):
        """
        Returns the boundary points of the domain at which the BC residuals
        will be evaluated on the next iteration of training. This function is
        used when imposing periodic boundary conditions.

        ARGUMENTS

            None required by this class. The error based sampling optimizer
            is the only class that needs arguments for this function.

        RETURNS

            bndry_pts   :       (P, self.d+1)-Torch tensor. Points at which the
                                  periodic BC residuals will be evaluated.
                                  These are pairs
                                        bndry_pts[i] = [t[i]; x_bndry[i]]

            opp_pts     :       (P, self.d+1)-Torch tensor. Points at which are
                                  on the boundary opposite from 'bndry_pts'
                                  These are pairs
                                        opp_pts[i] = [t[i]; x_opp[i]],
                                  where position x_opp[i] is on the opposite
                                  boundary from position x_bndry[i].
                                For instance, in the 1d case, if
                                    x_bndry[i] = 0 then x_opp[i] = 1.
        """
        return (
            self.mesh_bndry[:self.n_bc_pts//2],
            self.mesh_bndry[self.n_bc_pts//2:]
        )


    def next_interior_points(self, *args, **kwargs):
        """
        Returns the interior points of the domain at which the PDE residuals
        will be evaluated on the next iteration of training

        ARGUMENTS

            None required by this class. The error based sampling optimizer
            is the only class that needs arguments for this function.

        RETURNS

            interior_pts   :   (# points, self.d+1)-Torch tensor. Points at
                                which the PDE residuals will be evaluated.
                                These are pairs interior_pts[i] = [t[i]; x[i]]
        """
        return self.mesh_interior


"""
--------------------------------------------------------------------------------
    Mesh Free Optimizer

TODOs:
- Error based sampling method, with parameter to control exploration /
  exploitation
- Sampling distribution that depends on time
--------------------------------------------------------------------------------
"""
# Sampling Imports
from .sampling import *

class MeshFreeOptimizer(PINN_Optimizer):
    """
    ----------------------------------------------------------------------------
        Setup functions
    ----------------------------------------------------------------------------
    """
    def __init__(self, n_pde_pts=None, n_bc_pts=None, n_ic_pts=None, d=1, T=1,
            error_based=False, p_exploit=0.5
        ):
        """
        ARGUMENTS

            n_pde_pts    :  Number of interior mesh points used to compute
                            the PDE loss on each iteration
                            Default: 2^(d+1), where d is # of spatial dimensions

            n_bc_pts     :  Number of boundary points used to compute the
                            boundary condition loss on each iterations
                            Default: 2^d

            n_ic_pts     :  Number of points used to the compute the initial
                            condition loss on each iterations
                            Default: 2^d

            d        :  Int. Spatial dimension
                        Default: 1
                        NOTE: Only works for d=1 currently

            T        :  Float. Sets time domain to [0,T]
                        Default: 1

            error_based   :     Flag. Set to True to perform error based
                                sampling
                                Default: False

            p_exploit     :     Float in [0,1]. Controls the proportion of
                                points saved from the previous training
                                iterations, based on their error
                                Default: 0.5
        """
        self.d = d
        self.T = T

         # Number of pde / boundary/ t0 training points
        if n_pde_pts is None:
            n_pde_pts = 2 ** (d+1)
        if n_bc_pts is None:
            n_bc_pts = 2 ** d
        if n_ic_pts is None:
            n_ic_pts = 2 ** d
        self.n_pde_pts = n_pde_pts
        self.n_bc_pts = n_bc_pts
        self.n_ic_pts = n_ic_pts

        self.t0_pts = None
        self.bndry_pts = None
        self.bndry_pairs = None
        self.interior_pts = None

        # Error based sampling parameters
        self.error_based = error_based
        self.p_exploit = p_exploit
        self.res_ic  = None   # Variables to store the
        self.res_bc  = None   # residuals associated with
        self.res_pde = None   # each region of the domain


    def setup(self, f_ic, f_bc):
        """
        Sets up various data structures used by the optimizer during the
        training loop (self.train())
        """
        # Save the functions for computing the initial condition and
        # boundary condition.
        # We will need these later, specifically in 'next_t0_condition'
        # and 'next_boundary_condition'
        self.f_ic = f_ic
        self.f_bc = f_bc

    """
    ----------------------------------------------------------------------------
        Domain Sampling Functions
    ----------------------------------------------------------------------------
    """
    def sampling_helper(self, n_samples, curr_samples, residuals, sampling_method):
        """
        General idea
        - given list of existing elements and their associated errors
        - select the ones with the highest error
        - generate the remaining points
        """
        # If 'samples' is None or error based sampling is disabled,
        # simply generate a fresh sample
        if curr_samples is None or not self.error_based or self.p_exploit <= 0:
            return sampling_method(n_samples, self.d, self.T)

        # Select the points with the highest error
        n_exploit = int(self.p_exploit * n_samples)
        exploit_inds = torch.topk(
            torch.abs(residuals), n_exploit, dim=0
        ).indices.view(-1)

        # Generate the remaining points
        new_samples = sampling_method(n_samples - n_exploit, self.d, self.T)

        # Combine the exploited and explore points
        # Case 1: Pairs of points from PBC
        if isinstance(curr_samples, tuple):
            combined_samples = tuple(
                torch.vstack((curr[exploit_inds], new))
                for curr, new in zip(curr_samples, new_samples)
            )
        # Case 2: Tensors
        else:
            combined_samples = torch.vstack(
                (curr_samples[exploit_inds].detach(), new_samples)
            )

        # Return
        return combined_samples


    def next_t0_condition(self, *args, **kwargs):
        """
        Returns a tuple of data used for evaluating the initial condition
        residuals on the next iteration of training

        ARGUMENTS

            None required by this class. The error based sampling optimizer
            is the only class that needs arguments for this function.

        RETURNS

            t0_points       :   (# points, self.d+1)-Torch tensor. Points at
                                which the initial condition is to be evaluated.
                                These are pairs t0_points[i] = [t[i]; x[i]]

            u_exact_t0      :   (# points, 1)-Torch tensor. Initial condition at
                                t=t[i], x=x[i]
        """
        # Generate the sample
        self.t0_pts = self.sampling_helper(
            self.n_ic_pts, self.t0_pts, self.res_ic,
            sampling_method=sample_t0_points
        )

        # Compute and return the full initial condition
        self.u_exact_t0 = self.f_ic(self.t0_pts)
        return self.t0_pts, self.u_exact_t0


    def next_boundary_condition(self, *args, **kwargs):
        """
        Returns a tuple of data used for evaluating the boundary condition
        residuals on the next iteration of training.

        ARGUMENTS

            None required by this class. The error based sampling optimizer
            is the only class that needs arguments for this function.

        RETURNS

            t0_points       :   (# points, self.d+1)-Torch tensor. Points at
                                which the initial condition is to be evaluated.
                                These are pairs t0_points[i] = [t[i]; x[i]]

            u_exact_t0      :   (# points, 1)-Torch tensor. Initial condition at
                                t=t[i], x=x[i]
        """
        # Verify this function isn't being called for periodic boundary conditions
        if self.f_bc is None:
            raise ValueError(
                'f_bc should not be None when next_boundary_condition() is called'
            )

        # Generate the sample
        self.bndry_pts = self.sampling_helper(
            self.n_bc_pts, self.bndry_pts, self.res_bc,
            sampling_method=sample_boundary_points
        )

        # Compute and return the full initial condition
        self.u_exact_bndry = self.f_ic(self.bndry_pts)
        return self.bndry_pts, self.u_exact_bndryW


    def next_boundary_pairs(self, *args, **kwargs):
        """
        Returns the boundary points of the domain at which the BC residuals
        will be evaluated on the next iteration of training. This function is
        used when imposing periodic boundary conditions.

        ARGUMENTS

            None required by this class. The error based sampling optimizer
            is the only class that needs arguments for this function.

        RETURNS

            bndry_pts   :       (P, self.d+1)-Torch tensor. Points at which the
                                  periodic BC residuals will be evaluated.
                                  These are pairs
                                        bndry_pts[i] = [t[i]; x_bndry[i]]

            opp_pts     :       (P, self.d+1)-Torch tensor. Points at which are
                                  on the boundary opposite from 'bndry_pts'
                                  These are pairs
                                        opp_pts[i] = [t[i]; x_opp[i]],
                                  where position x_opp[i] is on the opposite
                                  boundary from position x_bndry[i].
                                For instance, in the 1d case, if
                                    x_bndry[i] = 0 then x_opp[i] = 1.
        """
        # Verify this function isn't being called for non periodic boundary
        # conditions
        if self.f_bc is not None:
            raise ValueError(
                'f_bc should be None when next_boundary_pairs() is called'
            )

        # Otherwise, sample pairs of points from the boundary
        self.bndry_pairs = self.sampling_helper(
            self.n_bc_pts, self.bndry_pairs, self.res_bc,
            sampling_method=sample_boundary_pairs
        )
        return self.bndry_pairs


    def next_interior_points(self, *args, **kwargs):
        """
        Returns the interior points of the domain at which the PDE residuals
        will be evaluated on the next iteration of training

        ARGUMENTS

            None required by this class. The error based sampling optimizer
            is the only class that needs arguments for this function.

        RETURNS

            interior_pts   :   (# points, self.d+1)-Torch tensor. Points at
                                which the PDE residuals will be evaluated.
                                These are pairs
                                        interior_pts[i] = [t[i]; x[i]]
        """
        # Sample points from the interior of the domain
        self.interior_pts = self.sampling_helper(
            self.n_pde_pts, self.interior_pts, self.res_pde,
            sampling_method=sample_interior_points
        )
        self.interior_pts.requires_grad = True
        return self.interior_pts

    """
    ----------------------------------------------------------------------------
        Training Functions
    ----------------------------------------------------------------------------
    """

    def step(self, res_ic, res_bc, res_pde):
        """
        This is the function called at the end of each iteration in the training
        loop. It allows the optimizer to base it's samples on the residuals on
        the different regions of the mesh.
        """
        self.res_ic = res_ic
        self.res_bc = res_bc
        self.res_pde = res_pde
