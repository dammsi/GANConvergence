# Zunaechst fur DiracGAN probieren.

import torch
from torch import optim
from attrdict import AttrDict
import os
from torch.optim import lr_scheduler
import numpy as np
from tqdm import tqdm

from data import latent_data
from data import target_data
from optimal_discriminator import get_optimal_disc # for optimal discriminator dynamics
from visualize import visualize_training

from model import Generator
from model import Discriminator
from train import trainer

import time
import matplotlib

import torchvision.utils as vutils
#from gan_eval_metrics import mnist_inception_score
from torch import autograd
from matplotlib import gridspec
# from lib import utils
#from lib.linalg import JacobianVectorProduct
import scipy.sparse.linalg as linalg

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------- Specify parameters for data, model and training -------------- #
param = {'device': 'cpu',
         'target_dim': 'dirac',  # 'dirac', '1D' or '2D'
         'latent_distribution': 'dirac',  # 'dirac', 'uniform', 'gaussian'
         'data_bias': 0.0,
         'z_dim': 1,  # dimension of latent_distribution
         'gan_type': 'wgan',  # 'wgan' or 'nsgan'
         'n_disc_train': 1,  # disc updates per generator update
         'lr_disc': 0.01,
         'lr_gen': 0.01,
         'n_epochs': 1000,
         'n_epochs_pic': 100,
         'batch_size': 1,  # should be 1 for 'dirac'
         'regularizer': 'wgan-lp', # 'off', 'gp', 'wgan-gp', 'wgan-lp', 'wgan-alp'
         'pen_weight': 1,  # penalty weight (lambda) for 'wgan'
         'add_penalty': False,
         'compare': 'off',  # 'off', optimal model ('optimal', only for disc_depth==1) or adaptive lipschitz ('wgan-alp')
         'schedule_lr': False  # whether to apply learning rate scheduler (inside training)
         }

# make parameters nicely accessible (via param._)
param = AttrDict(param)

# ----------------- Initialize the models with optimizers ----------------- #
# initialize models for regular training
disc = Discriminator(param)
gen = Generator(param)

# initialize generator parameter
with torch.no_grad():
    gen.gen[0].weight.fill_(-1.5)
    disc.disc[0].weight.fill_(-0.3)

disc_optimizer = optim.SGD(disc.parameters(), lr=param.lr_disc)
gen_optimizer = optim.SGD(gen.parameters(), lr=param.lr_gen)

# initialize models for optimal training
if param.compare != 'off':
    compare_disc = Discriminator(param)
    compare_gen = Generator(param)

    # initialize generator like the normal generator
    with torch.no_grad():
        compare_gen.gen[0].weight.fill_(-1.55)
        compare_disc.disc[0].weight.fill_(-0.3)

    compare_disc_optimizer = optim.SGD(compare_disc.parameters(), lr=param.lr_disc)
    compare_gen_optimizer = optim.SGD(compare_gen.parameters(), lr=param.lr_gen)

# make directory for saving figures.
if param.compare == 'off':
    param['save_dir'] = f"Test_Path_Angle/{param.gan_type}"
else:
    param['save_dir'] = f"Test_Path_Angle/{param.gan_type}_{param.compare}"

if not os.path.exists(param.save_dir):
    os.makedirs(param.save_dir)

if not os.path.exists(os.path.join(param.save_dir, 'checkpoints')):
    os.makedirs(os.path.join(param.save_dir, 'checkpoints'))
torch.save({'state_gen': gen.state_dict(), 'state_dis': disc.state_dict()},
              os.path.join(param.save_dir, 'checkpoints/%i.state'%(0)))

# ---------------------------- Train the model ---------------------------- #
if param.compare == 'off':
    trainer(disc=disc,
            gen=gen,
            disc_optimizer=disc_optimizer,
            gen_optimizer=gen_optimizer,
            param=param
            )

torch.save({'state_gen': gen.state_dict(), 'state_dis': disc.state_dict()},
              os.path.join(param.save_dir, 'checkpoints/%i.state'%(1)))

checkpoint_1 = torch.load(os.path.join(param.save_dir, 'checkpoints/%i.state'%(0))))
checkpoint_2 = torch.load(os.path.join(param.save_dir, 'checkpoints/%i.state'%(1))))

def compute_path_stats(gen, dis, checkpoint_1, checkpoint_2, dataloader, config,
                       model_loss_gen, model_loss_dis,
                       device=None, path_min=-0.1, path_max=1.1, n_points=100,
                       key_gen='state_gen', key_dis='state_dis', verbose=False):
    """
    Computes stats for plotting path between checkpoint_1 and checkpoint_2.

    Parameters
    ----------
    gen: Generator
    dis: Discriminator
    checkpoint_1: pytorch checkpoint
        first checkpoint to plot path interpolation
    checkpoint_2: pytorch checkpoint
        second checkpoint to plot path interpolation
    dataloader: pytorch DataLoader
        real data loader (mnist)
    config: Namespace
        configuration (hyper-parameters) for the generator/discriminator
    model_loss_dis, model_loss_gen: function
        returns generator and discriminator losses given the discriminator output
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We compute diff which is a vector representing the vector between input1 and input2
    # it is useful later when we compute the cosine similarity and dot product.
    params_diff = []
    for name, p in gen.named_parameters():
        d = (checkpoint_1[key_gen][name] - checkpoint_2[key_gen][name])
        params_diff.append(d.flatten())

    for name, p in dis.named_parameters():
        d = (checkpoint_1[key_dis][name] - checkpoint_2[key_dis][name])
        params_diff.append(d.flatten())

    params_diff = torch.cat(params_diff)

    # The different statistics we want to compute are saved in a dict.
    hist = {'alpha': [], 'cos_sim': [], 'dot_prod': [], 'gen_loss': [], 'dis_loss': [],
            'penalty': [], 'grad_gen_norm': [], 'grad_dis_norm': [], 'grad_total_norm': []}

    start_time = time.time()

    # Compute statistics we are interested in for different values of alpha.
    for alpha in np.linspace(path_min, path_max, n_points):

        ############### Computing and loading interpolation ##############
        # We compute the interpolation between input1 and input2
        # with interpolation-coefficient = alpha and load them into the model.
        # When alpha = 0 then the model is equal to the parameters of input1.
        state_dict_gen = gen.state_dict()
        for p in checkpoint_1[key_gen]:
            state_dict_gen[p] = alpha * checkpoint_2[key_gen][p] + (1 - alpha) * checkpoint_1[key_gen][p]
        gen.load_state_dict(state_dict_gen)

        state_dict_dis = dis.state_dict()
        for p in checkpoint_1[key_dis]:
            state_dict_dis[p] = alpha * checkpoint_2[key_dis][p] + (1 - alpha) * checkpoint_1[key_dis][p]
        dis.load_state_dict(state_dict_dis)

        gen = gen.to(device)
        dis = dis.to(device)
        #################################################################

        ######### Compute Loss and Gradient over Full-Batch ##########
        # cos_sim = 0
        # norm_grad_gen = 0
        # norm_grad_dis = 0
        # dot_prod = 0

        gen_loss_epoch = 0
        dis_loss_epoch = 0
        penalty_epoch = 0
        grad_gen_epoch = {}
        for name, param in gen.named_parameters():
            grad_gen_epoch[name] = torch.zeros_like(param).flatten()
        grad_dis_epoch = {}
        for name, param in dis.named_parameters():
            grad_dis_epoch[name] = torch.zeros_like(param).flatten()

        n_data = 0
        t0 = time.time()
        for i, x_true in enumerate(dataloader):
            x_true = x_true[0]
            z = torch.randn(x_true.size(0), config.nz, 1, 1)

            x_true = x_true.to(device)
            z = z.to(device)

            for p in gen.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            for p in dis.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            ################# Compute Loss #########################
            # TODO: Needs to be changed to be able to handle different kind of loss
            x_gen = gen(z)
            dis_loss, _, _ = model_loss_dis(x_true, x_gen.detach(), dis, device)
            gen_loss, _ = model_loss_gen(x_gen, dis, device)
            if config.model == 'wgan_gp':
                penalty = dis.get_penalty(x_true.detach(), x_gen.detach()).mean()
                dis_loss += config.gp_lambda * penalty
            else:
                penalty = torch.zeros(1)
            #################################################

            for p in dis.parameters():
                p.requires_grad = False
            gen_loss.backward(retain_graph=True)
            for p in dis.parameters():
                p.requires_grad = True

            for p in gen.parameters():
                p.requires_grad = False
            dis_loss.backward()
            for p in gen.parameters():
                p.requires_grad = True

            for name, param in gen.named_parameters():
                grad_gen_epoch[name] += param.grad.flatten() * len(x_true)

            for name, param in dis.named_parameters():
                grad_dis_epoch[name] += param.grad.flatten() * len(x_true)

            gen_loss_epoch += gen_loss.item() * len(x_true)
            dis_loss_epoch += dis_loss.item() * len(x_true)
            penalty_epoch += penalty.item() * len(x_true)
            n_data += len(x_true)
        ########################################################

        gen_loss_epoch /= n_data
        dis_loss_epoch /= n_data
        penalty_epoch /= n_data

        grad_gen = []
        for name, _ in gen.named_parameters():
            grad_gen.append(grad_gen_epoch[name])
        grad_dis = []
        for name, param in dis.named_parameters():
            param_flat = param.flatten()
            grad_param = grad_dis_epoch[name]
            if config.model == 'wgan':
                # zero-out gradient that violate wgan weight constraints
                zero_mask = (torch.abs(param_flat) == config.clip) &\
                            (torch.sign(grad_param) == torch.sign(param_flat))
                grad_param[zero_mask] = 0.0
            grad_dis.append(grad_param)

        grad_gen = torch.cat(grad_gen) / n_data
        grad_dis = torch.cat(grad_dis) / n_data
        grad_all = torch.cat([grad_gen, grad_dis])

        ####### Compute statistics we are interested in ##########
        # Compute squared norm of the gradient
        norm_grad_gen = (grad_gen**2).sum().cpu().numpy()
        norm_grad_dis = (grad_dis**2).sum().cpu().numpy()

        # Compute the dot product (unnormalized cosine similarity)
        dot_prod = (grad_all * params_diff).sum() / torch.sqrt((params_diff**2).sum())

        # Compute cosine similarity
        cos_sim = dot_prod / torch.sqrt((grad_all**2).sum())

        dot_prod = dot_prod.item()
        cos_sim = cos_sim.item()

        # # Compute cosine similarity
        # cos_sim = 1 - distance.cosine(grad_all, params_diff)

        # # Compute the dot product (unnormalized cosine similarity)
        # dot_prod = (grad_all * params_diff).sum() / np.sqrt((params_diff**2).sum())
        ##########################################################
        if verbose:
            print("Alpha: %.2f, Angle: %.2f, Generator loss: %.2e, Discriminator loss: %.2e, Penalty: %.2f, Gen grad norm: %.2e, Dis grad norm: %.2e, Time: %.2fsec"
                  % (alpha, cos_sim, gen_loss_epoch, dis_loss_epoch, penalty_epoch, norm_grad_gen, norm_grad_dis, time.time() - t0))

        hist['alpha'].append(alpha)
        hist['cos_sim'].append(cos_sim)
        hist['dot_prod'].append(dot_prod)
        hist['gen_loss'].append(gen_loss_epoch)
        hist['dis_loss'].append(dis_loss_epoch)
        hist['penalty'].append(penalty_epoch)
        hist['grad_gen_norm'].append(norm_grad_gen)
        hist['grad_dis_norm'].append(norm_grad_dis)
        hist['grad_total_norm'].append(norm_grad_dis + norm_grad_gen)

    if verbose:
        print("Time to finish: %.2f minutes" % ((time.time() - start_time) / 60.))

    return hist
