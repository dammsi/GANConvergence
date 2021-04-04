# -------------- Train GANs with different regularizers in 1D -------------- #

import os
import torch
from attrdict import AttrDict
from torch import optim

from model import Discriminator
from model import Generator
from train import trainer
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
         'n_hidden': 20,  # number of hidden neurons if disc_depth==2
         'n_disc_train': 1,  # disc updates per generator update
         'lr_disc': 0.01,
         'lr_gen': 0.01,
         'n_epochs': 5000,
         'n_epochs_pic': 500,
         'n_epochs_loss': 50,
         'batch_size': 50,
         'regularizer': 'wgan-gp',  # 'off', 'gp', 'wgan-gp', 'wgan-lp',...
         'pen_weight': 1.,  # penalty weight (lambda) for 'wgan'
         'schedule_lr': False,  # whether to apply learning rate scheduler (inside training)
         'n_runs': 10  # 'None' or int
         }

# make parameters nicely accessible (via param._)
param = AttrDict(param)

# make directory for saving figures
param['save_dir'] = f"E4/1D/{param.gan_type}_{param.regularizer}_LR1_{param.pen_weight}"

if not os.path.exists(param.save_dir):
    os.makedirs(param.save_dir)

# ----------------- Initialize the models with optimizers ----------------- #
disc = Discriminator(param)
gen = Generator(param)

disc_optimizer = optim.SGD(disc.parameters(), lr=param.lr_disc)
gen_optimizer = optim.SGD(gen.parameters(), lr=param.lr_gen)

# ---------------------------- Single runs ---------------------------- #
if param.n_runs is None:
    # -------------------- Initialize weights of the models -------------------- #
    # initialize generator parameter
    with torch.no_grad():
        gen.gen[0].weight.fill_(0.5)
        gen.gen[0].bias.fill_(1.5)

    # set bias = 0 for wgan (no effect, for visualization purposes)
    with torch.no_grad():
        # gen.gen[0].bias.fill_(-5)
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
else:
    W_losses = {}
    save_dir_orig = param['save_dir']
    for run in range(param.n_runs):
        # set up new directory
        param['save_dir'] = save_dir_orig + f"/run_{run}"
        if not os.path.exists(param.save_dir):
            os.makedirs(param.save_dir)

        # -------------------- Initialize weights of the models -------------------- #
        # initialize generator parameter
        with torch.no_grad():
            gen.gen[0].weight.fill_(0.5)
            gen.gen[0].bias.fill_(1.5)

        # set bias = 0 for wgan (no effect, for visualization purposes)
        with torch.no_grad():
            # gen.gen[0].bias.fill_(-5)
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

saving(param, disc, gen)