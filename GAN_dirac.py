# -------- Train the DiracGAN with different objectives & regularizers -------- #

import torch
from torch import optim
from attrdict import AttrDict
import os
from itertools import product

from model import Generator
from model import Discriminator
from train import trainer
from visualize import plot_multiple_runs
from postprocess import saving

# -------------- Specify parameters for data, model and training -------------- #
param = {'device': 'cpu',
         'target_dim': 'dirac',  # 'dirac', '1D' or '2D'
         'latent_distribution': 'dirac',  # 'dirac', 'uniform', 'gaussian'
         'dirac_noise': True,  # 'True' or 'False'
         'data_bias': 0,
         'z_dim': 1,  # dimension of latent_distribution
         'gan_type': 'wgan',  # 'wgan' or 'nsgan' or 'vanilla'
         'n_disc_train': 1,  # disc updates per gen update
         'lr_disc': 0.01,
         'lr_gen': 0.01,
         'n_epochs': 1000,
         'n_epochs_pic': 200,
         'n_epochs_loss': 5,
         'batch_size': 10,
         'dataset_size': 100,
         'regularizer': 'gp',  # 'off', 'gp', 'wgan-gp', 'wgan-lp', 'wgan-alp'
         'pen_weight': 1.0,  # penalty weight (lambda) for 'wgan'
         'schedule_lr': False,  # whether to apply learning rate scheduler (inside training)
         'run_type': 'experiment',  # 'single_run', 'multi_run' or 'experiment'
         'n_runs': 1  # for multi_run and experiment
         }

# make parameters nicely accessible (via param._)
param = AttrDict(param)
# make directory for saving
param['save_dir'] = f"Dirac/{param.gan_type}_{param.regularizer}_{param.pen_weight}"

# ----------------- Initialize the models with optimizers ----------------- #
# initialize models for regular training
disc = Discriminator(param)
gen = Generator(param)

disc_optimizer = optim.SGD(disc.parameters(), lr=param.lr_disc)
gen_optimizer = optim.SGD(gen.parameters(), lr=param.lr_gen)

# ---------------------------- Single runs ---------------------------- #
if param.run_type == 'single_run':
    if not os.path.exists(param.save_dir):
        os.makedirs(param.save_dir)

    # ------------- Initialize optimizers & weights of the models ------------- #
    disc_optimizer = optim.SGD(disc.parameters(), lr=param.lr_disc)
    gen_optimizer = optim.SGD(gen.parameters(), lr=param.lr_gen)

    with torch.no_grad():
        gen.gen[0].weight.fill_(-1.)
        disc.disc[0].weight.fill_(0.1)

    # ---------------------------- Train the model ---------------------------- #
    trainer(disc=disc,
            gen=gen,
            disc_optimizer=disc_optimizer,
            gen_optimizer=gen_optimizer,
            param=param
            )

# ---------------------------- Experiments ---------------------------- #
elif param.run_type == 'experiment':
    name_1 = 'lr_disc'
    values_1 = [0.01]
    name_2 = 'pen_weight'
    values_2 = [0.3, 0.7, 1.0]

    save_dir_orig = param['save_dir']
    for specification in product(values_1, values_2):
        current_1 = specification[0]
        current_2 = specification[1]

        print(f"Starting with {name_1}: {current_1}, {name_2}: {current_2}")
        # update current parameters
        param[name_1] = current_1
        param[name_2] = current_2

        # re-initialize optimizers for changed LRs
        disc_optimizer = optim.SGD(disc.parameters(), lr=param.lr_disc)
        gen_optimizer = optim.SGD(gen.parameters(), lr=param.lr_gen)

        current_save_dir = save_dir_orig + f"/{name_1}{current_1}_{name_2}{current_2}"
        W_losses = {}
        for run in range(param.n_runs):
            # set up new directory
            param['save_dir'] = current_save_dir + f"/run_{run + 1}"
            if not os.path.exists(param.save_dir):
                os.makedirs(param.save_dir)

            # -------------------- Initialize weights of the models -------------------- #
            # initialize parameter
            with torch.no_grad():
                gen.gen[0].weight.fill_(-1.)
                disc.disc[0].weight.fill_(0.0)

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
