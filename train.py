# ----- TRAIN ----- #

import torch
from torch.optim import lr_scheduler
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import tikzplotlib

from data import latent_data
from data import target_data
from data import get_target_samples
from visualize import visualize_training
from sinkhorn_loss import sinkhorn_loss

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "xelatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


# LOSSES #

def discriminator_loss(disc, gen, true_samples, gen_samples, param):
    """ todo
    Calculate discriminator loss for NSGAN and WGAN with regularization techniques.
    Note that the discriminator maximizes the value function, such that it minimizes the
    negative value function (minimizing the loss).
    :parameter:
        - disc : Discriminator instance
        - gen : Generator instance
        - true_samples : torch.tensor
            Samples from data.target_distribution()
        - noise : torch.tensor
               Noise from data.latent_data()
        - param : AttrDict
            Dictionary containing additional parameters, e.g., gan_type, regularizer and pen_weight
    """
    gen_samples = gen_samples.detach()

    if param.gan_type == "nsgan" or param.gan_type == "vanilla":
        # D maximizes: phi(D(0)) + phi(1-D(G(1))) with phi = log
        return - (torch.mean(torch.log(disc(true_samples))) + torch.mean(torch.log(1 - disc(gen_samples))))

    elif param.gan_type == "wgan":
        # --- Gradient & Lipschitz Penalties --- #
        # 1. Sampling Technique
        n_grad_samples = 1  # number of samples to approximate expectation over penalty distribution
        #x_hat = target_data(param)[0:n_grad_samples]  # sample from target distribution
        x_hat = gen_samples[0:n_grad_samples]   # x_hat from gen_samples
        x_hat.requires_grad_(True)

        # 2. Calculate Gradient
        grad = torch.autograd.grad(
            inputs=x_hat,
            outputs=disc(x_hat),
            grad_outputs=torch.ones_like(disc(x_hat)),
            create_graph=True
        )

        # 3. Calculate penalties
        # 3.1 Gradient Penalty (simple GP)
        if param.regularizer == 'gp':
            penalty = torch.norm(grad[0]) ** 2
        # 3.2 Gradient Penalty for Wasserstein GAN (WGAN-GP)
        if param.regularizer == 'wgan-gp':
            penalty = (torch.norm(torch.mean(grad[0])) - 1) ** 2
        # 3.3 Lipschitz Penalty for Wasserstein GAN (WGAN-LP)
        if param.regularizer == 'wgan-lp':
            penalty = (torch.max(torch.zeros(1), torch.norm(torch.mean(grad[0])) - 1)) ** 2
        # D maximizes: phi(D(true_samples)) + phi(1-D(G(noise))) with phi = id
        if param.regularizer == 'off':
            return - (torch.mean(disc(true_samples)) - torch.mean(disc(gen_samples)))
        else:
            return - (torch.mean(disc(true_samples)) - torch.mean(disc(gen_samples))) + param.pen_weight * penalty

    else:
        raise ValueError("gan_type should be 'nsgan', 'vanilla' or 'wgan'.")


def generator_loss(disc, gen, noise, param):
    """ todo
    Calculate generator loss according to generalized GAN loss.
    :parameter:
        - disc : Discriminator instance
        - gen : Generator instance
        - noise : torch.tensor
            Noise from data.latent_data()
        - param : AttrDict
            Dictionary containing additional parameters, e.g., gan_type
    """
    if param.gan_type == "wgan":
        # G minimizes: phi(1-D(G(1))) with phi = id
        return - torch.mean(disc(gen(noise)))

    elif param.gan_type == "nsgan":
        # G minimizes: - phi(D(G(1))) with phi = log
        return - torch.mean(torch.log(disc(gen(noise))))

    elif param.gan_type == "vanilla":
        # G minimizes: - phi(1-D(G(1))) with phi = log
        return torch.mean(torch.log(1 - disc(gen(noise))))

    else:
        raise ValueError("gan_type should be 'nsgan', 'vanilla' or 'wgan'.")


def init_weights(net):
    """
    Small subroutine to initalize weights following Xavier_Uniform.
    :param net: Network (either Discriminator or Generator instance)
    :return:
    """
    if type(net) == torch.nn.Linear:
        # use xavier_uniform (aka Glorot) initialization with scaling parameter gain
        torch.nn.init.xavier_uniform_(net.weight, gain=1.5)

# TRAIN #

def trainer(disc, gen, disc_optimizer, gen_optimizer, param,
            compare_disc=None, compare_gen=None, compare_gen_optimizer=None, compare_disc_optimizer=None):
    """
    Train a GAN with given agents and optimizers according to the specified arguments in param.
    This function requires the two functions target_data and latent_data to sample data from,
    as well as the functions disc_loss and gen_loss to calculate the losses.
    :parameter:
        - disc : Discriminator instance
        - gen : Generator instance
        - disc_optimizer : torch.optimizer
            Optimizer for the Discriminator
        - gen_optimizer : torch.optimizer
            Optimizer for the Generator
        - param : AttrDict
            Dictionary containing additional parameters, e.g., target_dim, gan_type and n_epochs
    """

    # --- Prepare for training & visualization --- #
    gen_losses = []
    disc_losses = []
    W_dist = []
    decision_boundaries = []

    # prepare trajectory positions for visualization
    if param.target_dim == "dirac":
        trajectory_points = np.array([gen.params().theta, disc.params().psi])
    elif param.target_dim == "1D" and param.disc_depth == 1:
        trajectory_points = np.array([gen.params().theta,
                                      disc.params().psi,
                                      gen.params().bias,
                                      disc.params().bias])
    elif param.target_dim == "1D" and param.disc_depth == 2:
        trajectory_points = None

    elif param.target_dim == "2D":
        trajectory_points = None

    # initialize LR Scheduler
    if param.schedule_lr:
        # MultiStepLR
        # lr_steps = [i / 10 * param.n_epochs for i in range(1, 10)]
        # disc_scheduler = lr_scheduler.MultiStepLR(disc_optimizer, milestones=lr_steps, gamma=1.)
        # gen_scheduler = lr_scheduler.MultiStepLR(gen_optimizer, milestones=lr_steps, gamma=0.99)
        # LambdaLR
        lambda_disc = lambda epoch: np.max([1 - epoch / param.n_epochs, 1 / 500])
        lambda_gen = lambda epoch: np.max([1 - epoch / param.n_epochs, 1 / 1000])
        disc_scheduler = lr_scheduler.LambdaLR(gen_optimizer, lambda_disc)
        gen_scheduler = lr_scheduler.LambdaLR(gen_optimizer, lambda_gen)

    # create target dataset
    dataset = target_data(param, n_samples=param.dataset_size)

    # --- The Training Loop --- #
    for epoch in tqdm(range(param.n_epochs + 1)):
        for _ in range(param.n_disc_train):
            # --- Sample from target and latent distribution --- #
            true_samples = get_target_samples(param, dataset)
            noise = latent_data(param)
            gen_samples = gen(noise)

            # --- Update discriminator --- #
            # Zero out the gradients before backpropagation
            disc_optimizer.zero_grad()
            # calculate discriminator loss
            disc_loss = discriminator_loss(disc, gen, true_samples=true_samples, gen_samples=gen_samples, param=param)
            # Backpropagation
            disc_loss.backward(retain_graph=True)
            # Gradient Update
            disc_optimizer.step()

        # --- Update the generator --- #
        # Zero out the gradients before backpropagation
        gen_optimizer.zero_grad()
        # Calculate discriminator loss
        gen_loss = generator_loss(disc, gen, noise=noise, param=param)
        # Backpropagation
        gen_loss.backward(retain_graph=True)
        # Gradient Update
        gen_optimizer.step()  # Does the update

        # --- LR Schedule --- #
        if param.schedule_lr:
            disc_scheduler.step()
            gen_scheduler.step()

        # --- Save losses and Wasserstein Distance for visualization --- #
        if epoch % param.n_epochs_loss == 0:
            disc_losses.append(disc_loss.data.numpy())
            gen_losses.append(gen_loss.data.numpy())
            W_dist.append(sinkhorn_loss(true_samples, gen_samples, n=param.batch_size))

        # --- Visualize the current state --- #
        if epoch % param.n_epochs_pic == 0:
            # Save current point of trajectory
            if param.target_dim == "dirac":
                coordinates = np.array([gen.params().theta, disc.params().psi])
                trajectory_points = np.vstack((trajectory_points, coordinates))
            if param.target_dim == "1D" and param.disc_depth == 1:
                coordinates = np.array([gen.params().theta,
                                        disc.params().psi,
                                        gen.params().bias,
                                        disc.params().bias])
                trajectory_points = np.vstack((trajectory_points, coordinates))

            # Save current decision boundaries
            if param.target_dim == "1D" and param.disc_depth == 2:
                w = torch.reshape(disc.disc[0].weight, [param.n_hidden, ]).detach()
                b = disc.disc[0].bias.detach()
                decision_boundary = (- b / w).tolist()
                decision_boundaries.append(decision_boundary)

            # Visualize the training
            visualize_training(gen, disc, epoch=epoch, true_samples=true_samples, gen_samples=gen_samples, param=param,
                               W_dist=W_dist, disc_losses=disc_losses, gen_losses=gen_losses,
                               trajectory_points=trajectory_points, decision_boundaries=decision_boundaries)

    # --- After training: Plot and save Loss and Wasserstein Distance --- #
    # Losses
    plt.plot([param.n_epochs_loss * i for i in range(len(gen_losses))], gen_losses,
             color='red', label="Generator Loss")
    plt.plot([param.n_epochs_loss * i for i in range(len(disc_losses))], disc_losses,
             color='orange', label="Discriminator Loss")
    # if param.schedule_lr:
    #    plt.vlines(lr_steps, -5, 5, color='gray', alpha=0.4)
    plt.legend()
    plt.title(f"Loss: {param.gan_type}")
    # Save and close the figure
    plt.savefig(f"{param.save_dir}/0_Loss.png", dpi=200)
    plt.clf()

    # Wasserstein Distance
    plt.plot([param.n_epochs_loss * i for i in range(len(W_dist))], W_dist, color='blue', label="Wasserstein Distance")
    #plt.ylabel(r"$\widehat{W}_1(\mathbb{P},\mathbb{Q})$")
    plt.xlabel("epochs")
    # Save and close the figure
    plt.savefig(f"{param.save_dir}/W_Distance.png")
    tikzplotlib.save(f"{param.save_dir}/W_Distance.tex")
    plt.clf()
    return(W_dist)