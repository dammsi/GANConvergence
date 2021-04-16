# ------- Train GANs with different objectives & regularizers in 2D ------- #

from torch import optim
from attrdict import AttrDict
import os
import math
from itertools import product

from model import Generator
from model import Discriminator
from train import trainer
from train import init_weights
from visualize import plot_multiple_runs
from postprocess import saving

# -------------- Specify data, model and training parameters -------------- #
param = {'device': 'cpu',
         'target_dim': '2D',  # 'dirac', '1D' or '2D'
         'target_type_2D': 'points',  # 'interval', 'points', 'circle', 'square', 'swiss_roll',
         'scale_factor': 2,
         'data_bias': [0, 0],
         'rot_angle': -0.0 * math.pi,
         'latent_distribution': 'gaussian',  # 'dirac', 'uniform', 'gaussian'
         'z_dim': 2,  # dimension of latent_distribution
         'gan_type': 'wgan',  # 'wgan' or 'nsgan'
         'disc_depth': 3,  # depth of discriminator
         'gen_depth': 3,  # depth of generator
         'n_hidden': 40,  # number of hidden neurons if disc_depth==2
         'n_disc_train': 1,  # disc updates per generator update
         'lr_disc': 0.005,
         'lr_gen': 0.005,
         'n_epochs': 100000,
         'n_epochs_pic': 25000,
         'n_epochs_loss': 500,
         'dataset_size': 1000,
         'batch_size': 75,
         'regularizer': 'gp',  # 'off', 'wgan-gp', 'wgan-lp', 'wgan-alp'
         'pen_weight': 1.,  # penalty weight (lambda)
         'schedule_lr': False,  # whether to apply learning rate scheduler
         'run_type': 'experiment',  # 'single_run', 'multi_run' or 'experiment'
         'n_runs': 10  # for multi_run and experiment
         }
# convert param to AttrDict
param = AttrDict(param)
# make directory for saving
param['save_dir'] = f"2D_{param.target_type_2D}/{param.gan_type}_{param.regularizer}_{param.pen_weight}"

# ----------------- Initialize the models with optimizers ----------------- #
disc = Discriminator(param)
gen = Generator(param)

# ---------------------------- Single runs ---------------------------- #
if param.run_type == 'single_run':
    if not os.path.exists(param.save_dir):
        os.makedirs(param.save_dir)

    # ------------- Initialize optimizers & weights of the models ------------- #
    disc_optimizer = optim.SGD(disc.parameters(), lr=param.lr_disc)
    gen_optimizer = optim.SGD(gen.parameters(), lr=param.lr_gen)

    gen.apply(init_weights)
    disc.apply(init_weights)

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
        gen.apply(init_weights)
        disc.apply(init_weights)

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
    values_1 = [0.005]
    name_2 = 'pen_weight'
    values_2 = [0.5, 2.0, 5.0]

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
            gen.apply(init_weights)
            disc.apply(init_weights)

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
