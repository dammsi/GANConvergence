# -------------- Define different target distributions -------------- #

import math
import random

import torch
from sklearn.datasets import make_swiss_roll


def target_data(param, n_samples=None):
    """
    Draws samples from target distribution. Available target distributions are Dirac,
    interval in 1D and manifolds (interval, circle, square, swiss_roll) and mixture of Gaussians in 2D.
    Interval in 1D can be scaled and moved (with bias),
    interval in 2D can be scaled, moved (bias) and rotated (rot_angle).
    :parameter:
        - param : AttrDict
            Dictionary containing (besides others) the following parameter:
            - batch_size : int
                Specify the number of returned samples.
            - target_dim : str
                Return points interval in 1D or interval in 2D
            - scale_factor : float
                Valid for target_time "1D" or "2D", scaling the unit interval
            - bias : flat (1D) / float (2D)
                If target_dim=="Dirac": Adds scalar to centered dirac distribution
                If target_dim=="1D": Adds scalar to scaled interval
                If target_dim=="2D": Adds vector to scaled interval
            - rot_angle : float
                Only for target_dim="2D", specifies the rotation angle of the interval (mod 2pi)
            - device : str
                Specify device for torch tensors ("cpu" or "cuda")
    :return: tensor with batch_size samples from specified and transformed distribution
    """
    if n_samples is None:
        n_samples = param.batch_size

    if param.target_dim == "dirac":
        im_dim = 1
        if param.dirac_noise == True:
            noise = 0.1 * torch.randn((n_samples, im_dim), device=param.device)
            return torch.zeros((n_samples, im_dim), device=param.device) + noise + param.data_bias
        else:
            return torch.zeros((n_samples, im_dim), device=param.device) + param.data_bias

    if param.target_dim == "1D":
        im_dim = 1
        # scale and shift uniform distribution U[0,1]
        samples = param.scale_factor * torch.rand((n_samples, im_dim), device=param.device) + param.data_bias
        return samples

    if param.target_dim == "2D" and param.target_type_2D == 'interval':
        xs = param.scale_factor * torch.rand(n_samples, device=param.device)
        # ys = torch.zeros(n_samples)
        ys = torch.randn(n_samples) * 0.05
        samples = torch.stack([xs, ys], 1)
        rot_samples = torch.empty((n_samples, 2))
        rot_matrix = torch.tensor([[math.cos(param.rot_angle), -math.sin(param.rot_angle)],
                                   [math.sin(param.rot_angle), math.cos(param.rot_angle)]])
        for i, sample in enumerate(samples):
            rot_samples[i] = torch.matmul(rot_matrix, sample)

        return rot_samples + torch.tensor(param.data_bias)

    if param.target_dim == "2D" and param.target_type_2D == 'points':
        # points = torch.tensor([[-1., 1.], [0,1.42], [0,-1.42], [1.42,0], [-1.42,0], [1., -1.], [1., 1.], [-1., -1.]])  # 8 Gaussians
        points = torch.tensor([[0., 1.08], [1, 0.41], [-1, 0.41], [0.6, -0.9], [-0.6, -0.9]])  # 5 Gaussians
        points_batch = torch.index_select(points, 0, index=torch.randint(low=0, high=points.shape[0], size=[n_samples]))
        points_w_noise = points_batch + 0.1 * torch.randn(size=points_batch.shape)
        return points_w_noise

    if param.target_dim == "2D" and param.target_type_2D == 'circle':
        rangle = 2 * math.pi * torch.rand(n_samples, device=param.device)
        points_batch = torch.stack((torch.sin(rangle), torch.cos(rangle)), 1)
        points_batch += 0.05 * torch.rand(size=points_batch.shape)
        return points_batch

    if param.target_dim == "2D" and param.target_type_2D == 'swiss_roll':
        data, _ = make_swiss_roll(n_samples=n_samples)
        # omit the 3rd dimension and scale into [0,1] (and cast to float)
        data = 0.1 * data[:, [0, 2]] - 0.5
        points_batch = torch.tensor(data).float()
        return points_batch

    if param.target_dim == "2D" and param.target_type_2D == 'square':
        # define the four corners
        A = torch.tensor([0, 1.5])
        B = torch.tensor([1.5, 0])
        C = torch.tensor([0, -1.5])
        D = torch.tensor([-1.5, 0])

        samples = torch.empty(size=(n_samples, 2))
        for i in range(n_samples):
            t = torch.rand(1)
            sec = random.choices(['AB', 'BC', 'CD', 'DA'])[0]
            samples[i] = t * eval(sec[0]) + (1 - t) * eval(sec[1]) + 0.05 * torch.rand_like(samples[i])
        return samples

    else:
        raise ValueError("Please specify target_dim: 'dirac', '1D' or '2D'.")


def get_target_samples(param, dataset):
    """
    Return random mini-batch from dataset
    :param param: AttrDict with batch_size
    :return: batch of target samples
    """
    indices = torch.randperm(len(dataset))[:param.batch_size]
    return dataset[indices]


def latent_data(param, n_samples=None):
    """
    Return latent codes from specified distribution.
    :param param: AttrDict with batch_size and latent_distribution
    :param n_samples: int
    :return: latent codes as torch tensor
    """
    if n_samples is None:
        n_samples = param.batch_size

    if param.latent_distribution == "gaussian":
        return torch.randn((n_samples, param.z_dim), device=param.device)

    elif param.latent_distribution == "uniform":
        return torch.rand((n_samples, param.z_dim), device=param.device)

    elif param.latent_distribution == "dirac":  # latent code is just a constant
        return torch.ones((n_samples, param.z_dim), device=param.device)
    else:
        raise ValueError("Latent distribution should be 'gaussian', 'uniform' or 'dirac'.")