import torch

"""
--------------------------------------------------------------------------------
    Functions for computing derivatives of network output wrt input
--------------------------------------------------------------------------------
"""

def compute_u_v(v, u):
    """
    Computes u_v for the independent variable v and dependent variable u
    """
    return torch.autograd.grad(u, v, torch.ones_like(u), create_graph=True)[0]


def compute_u_vv(v, u_v):
    """
    Computes u_vv for the independent variable 'v' and dependent variable 'u'
        (not a parameter here)
    Must be given u_v, the partial derivative computed using compute_u_v()
    """
    return torch.autograd.grad(
        u_v, v, torch.ones_like(u_v), create_graph=True
    )[0]


"""
--------------------------------------------------------------------------------
    Transport
--------------------------------------------------------------------------------
"""
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
    return torch.cos(2 * torch.pi * s[:,1]).view(-1,1)


def u_exact_transport(v, a=1, f=None):
    """
    Exact solution to the transport equation

                    u_t + u_x = 0

    on (t,x) in [0,1] x [0,1] with the initial condition

                    u(0,x) = f(x)

    ARGUMENTS

        v    :   (# points, d+1)-Torch tensor of (t[i], x[i]) pairs where
                     v[i,0] = t[i] and s[i,1:] = x[i]

        a    :   (d,)-Torch tensor. The transport velocity
                 Default: a = [1, 1, ..., d]^t

        f    :   Callable or None. Function defining the initial condition
                 If None, the initial condition will be
                        u(0,x) = cos(2 pi x)

    RETURNS

        u   :   (# points, 1)-Torch tensor with u[i] = u(t[i], x[i])
    """
    # Handle the initial condition
    if f is None:
        f = lambda x: torch.cos(2 * torch.pi * x)
    # Compute the exact solution
    x_t = s[:,1] - s[:,0]
    return f(x_t).view(-1,1)
