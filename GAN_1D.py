# ------- Train GANs with different objectives & regularizers in 1D ------- #

import os
import torch
from attrdict import AttrDict
from torch import optim
from itertools import product

from model import Discriminator
from model import Generator
from train import trainer
from train import init_weights
from visualize import plot_multiple_runs
from postprocess import saving

# -------------- Specify data, model and training parameters -------------- #
param = {'device': 'cpu',
         'target_dim': '1D',  # 'dirac', '1D' or '2D'
         'scale_factor': 2,
         'data_bias': -2,
         'latent_distribution': 'uniform',  # 'dirac', 'uniform', 'gaussian'
         'z_dim': 1,  # dimension of latent_distribution
         'gan_type': 'wgan',  # 'wgan' or 'nsgan'
         'disc_depth': 2,  # depth of discriminator (1 or 2)
         'n_hidden': 15,  # number of hidden neurons if disc_depth==2
         'n_disc_train': 1,  # disc updates per generator update
         'lr_disc': 0.001,
         'lr_gen': 0.001,
         'n_epochs': 10000,
         'n_epochs_pic': 1000,
         'n_epochs_loss': 25,
         'dataset_size': 1000,
         'batch_size': 50,
         'regularizer': 'gp',  # 'off', 'wgan-gp', 'wgan-lp', 'gp'
         'pen_weight': 1.0,  # penalty weight (lambda)
         'schedule_lr': False,  # whether to apply learning rate scheduler
         'run_type': 'experiment',  # 'single_run', 'multi_run' or 'experiment'
         'n_runs': 10  # for multi_run and experiment
         }

# convert param to AttrDict
param = AttrDict(param)
# make directory for saving figures
param['save_dir'] = f"1D/{param.gan_type}_{param.regularizer}_{param.pen_weight}"

# ----------------- Initialize the models with optimizers ----------------- #
disc = Discriminator(param)
gen = Generator(param)

if not os.path.exists(param.save_dir):
    os.makedirs(param.save_dir)

# ---------------------------- Single runs ---------------------------- #
if param.run_type == 'single_run':
    if not os.path.exists(param.save_dir):
        os.makedirs(param.save_dir)

    # ------------- Initialize optimizers & weights of the models ------------- #
    disc_optimizer = optim.SGD(disc.parameters(), lr=param.lr_disc)
    gen_optimizer = optim.SGD(gen.parameters(), lr=param.lr_gen)

    with torch.no_grad():
        gen.gen[0].weight.fill_(0.5)
        gen.gen[0].bias.fill_(1.5)
        disc.apply(init_weights)
        # set bias = 0 for wgan (no effect, for visualization purposes)
        if param.gan_type == 'wgan':
            if param.disc_depth == 1:
                disc.disc[0].bias.fill_(0)
            if param.disc_depth == 2:
                disc.disc[2].bias.fill_(0)

    # ---------------------------- Train the model ---------------------------- #
    trainer(disc=disc,
            gen=gen,
            disc_optimizer=disc_optimizer,
            gen_optimizer=gen_optimizer,
            param=param
            )

# ---------------------------- Multiple runs ---------------------------- #
elif param.run_type == 'multi_run':
    if not os.path.exists(param.save_dir):
        os.makedirs(param.save_dir)

    W_losses = {}
    save_dir_orig = param['save_dir']

    disc_optimizer = optim.SGD(disc.parameters(), lr=param.lr_disc)
    gen_optimizer = optim.SGD(gen.parameters(), lr=param.lr_gen)

    for run in range(param.n_runs):
        # set up new directory
        param['save_dir'] = save_dir_orig + f"/run_{run + 1}"
        if not os.path.exists(param.save_dir):
            os.makedirs(param.save_dir)

        # -------------------- Initialize weights of the models -------------------- #
        with torch.no_grad():
            gen.gen[0].weight.fill_(0.5)
            gen.gen[0].bias.fill_(1.5)
            disc.apply(init_weights)
            # set bias = 0 for wgan (no effect, for visualization purposes)
            if param.gan_type == 'wgan':
                if param.disc_depth == 1:
                    disc.disc[0].bias.fill_(0)
                if param.disc_depth == 2:
                    disc.disc[2].bias.fill_(0)

        # ---------------------------- Train the model ---------------------------- #
        print(f"Starting run {run + 1}/{param.n_runs}...")
        W_losses[run] = trainer(disc=disc,
                                gen=gen,
                                disc_optimizer=disc_optimizer,
                                gen_optimizer=gen_optimizer,
                                param=param
                                )
    # return to original directory
    param['save_dir'] = save_dir_orig
    plot_multiple_runs(W_losses, param)

    # save parameters (and optionally the models)
    saving(param, disc, gen)

    # ---------------------------- Experiments ---------------------------- #
elif param.run_type == 'experiment':
    name_1 = 'lr_disc'
    values_1 = [0.001, 0.005, 0.01]
    name_2 = 'pen_weight'
    values_2 = [1.0]

    gen_optimizer = optim.SGD(gen.parameters(), lr=param.lr_gen)

    save_dir_orig = param['save_dir']
    for specification in product(values_1, values_2):
        current_1 = specification[0]
        current_2 = specification[1]

        print(f"Starting with {name_1}: {current_1}, {name_2}: {current_2}")
        # update current parameters
        param[name_1] = current_1
        param[name_2] = current_2

        # (re)initialize optimizers for (potentially) changed LRs
        disc_optimizer = optim.SGD(disc.parameters(), lr=param.lr_disc)

        current_save_dir = save_dir_orig + f"/{name_1}{current_1}_{name_2}{current_2}"
        W_losses = {}
        for run in range(param.n_runs):
            # set up new directory
            param['save_dir'] = current_save_dir + f"/run_{run + 1}"
            if not os.path.exists(param.save_dir):
                os.makedirs(param.save_dir)

            # -------------------- Initialize weights of the models -------------------- #
            with torch.no_grad():
                gen.gen[0].weight.fill_(0.5)
                gen.gen[0].bias.fill_(1.5)
                disc.apply(init_weights)
                # set bias = 0 for wgan (no effect, for visualization purposes)
                if param.gan_type == 'wgan':
                    if param.disc_depth == 1:
                        disc.disc[0].bias.fill_(0)
                    if param.disc_depth == 2:
                        disc.disc[2].bias.fill_(0)

            # ---------------------------- Train the model ---------------------------- #
            print(f"Starting run {run + 1}/{param.n_runs}...")
            W_losses[run] = trainer(disc=disc,
                                    gen=gen,
                                    disc_optimizer=disc_optimizer,
                                    gen_optimizer=gen_optimizer,
                                    param=param
                                    )
        # save parameters (and optionally the models) for each specification
        param['save_dir'] = current_save_dir
        saving(param, disc, gen)

        # return to original directory and plot the losses for all runs
        param['save_dir'] = save_dir_orig
        plot_multiple_runs(W_losses, param, name_1, name_2, current_1, current_2)
