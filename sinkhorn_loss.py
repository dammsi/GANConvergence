# Implementation of Sinkhorn Distance by Gabriel Peyré et al.
# https://github.com/gpeyre/SinkhornAutoDiff

import torch
from torch.autograd import Variable
import numpy as np
from attrdict import AttrDict
import math
from tqdm import tqdm

from data import target_data
from data import get_target_samples


def sinkhorn_loss(x, y, n, epsilon=0.01, niter=100):
    """-------
    Given two empirical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in Sinkhorn loop
    :param x: x-coordinates
    :param y: y-coordinates
    :param epsilon: regularization parameter
    :param n: number of samples from each distribution
    :param niter: number of maximum iterations
    :return: Sinkhorn loss
    """

    # The Sinkhorn algorithm takes as input three variables :
    C = Variable(cost_matrix(x, y))  # Wasserstein cost function

    # both marginals are fixed with equal weights
    # mu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    # nu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    mu = Variable(1. / n * torch.FloatTensor(n).fill_(1), requires_grad=False)
    nu = Variable(1. / n * torch.FloatTensor(n).fill_(1), requires_grad=False)

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10 ** (-1)  # stopping criterion

    # Elementary operations .....................................................................
    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        u1 = u  # useful to check the update
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        err = (u - u1).abs().sum()

        actual_nits += 1
        if (err < thresh).data.numpy():
            break
    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    # cost = torch.sum(pi * C)  # Sinkhorn cost
    cost = torch.sqrt(torch.sum(pi * C))  # Sinkhorn cost (square root)
    return cost


def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    return c


# estimate W_1 for comparing target distributions to themselves
def estimate_W1():
    # specify the target data
    param = {'device': 'cpu',
             'target_dim': '1D',  # 'dirac', '1D' or '2D'
             'target_type_2D': 'uniform',  # 'interval', 'points', 'circle', 'square', 'swiss_roll',
             'scale_factor': 2,
             'data_bias': -2,
             'dataset_size': 1000,
             'batch_size': 50
             }
    # convert param to AttrDict
    param = AttrDict(param)
    dataset = target_data(param=param, n_samples=param.dataset_size)
    W_est = []
    for i in tqdm(range(1000)):
        samples_1 = get_target_samples(param=param, dataset=dataset)
        samples_2 = get_target_samples(param=param, dataset=dataset)
        W_est.append(sinkhorn_loss(samples_1, samples_2, n=param.batch_size))
    print(np.array(W_est).mean())
    print(np.array(W_est).std())

# estimate_W1()
