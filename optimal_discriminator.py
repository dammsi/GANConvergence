# Optimal Discriminators
import numpy as np
import math
# from model import Generator
import torch

# from model import Discriminator
# from attrdict import AttrDict
# for 1D and ReLU activation

def get_optimal_disc(optimal_disc, gen, param):
    """
    Return the theoretically optimal discriminator for given true and fake data.
    :parameter
        - optimal_disc : Discriminator
            optimal discriminator instance
        - trues:
            For param.gan_type == '1D': list[left_end, right_end]
        - fakes:
            For param.gan_type == '1D': list[left_end, right_end]
        - param : AttrDict
            dictionary with additional parameters
    :return: optimal discriminator with updated parameters
    """
    # FIXME: What to do with nsgan and optimal disc??
    if param.target_dim == "dirac":
        theta_true = 0  # + param.bias
        theta_gen = gen.params().theta
        # optimal psi in {-1,1}
        opt_psi = math.copysign(1, theta_true - theta_gen)
        with torch.no_grad():
            optimal_disc.disc[0].weight.fill_(opt_psi)

    if param.target_dim == "1D":
        true_interval = [param.data_bias,
                         param.data_bias + param.scale_factor]
        gen_interval = [min(gen.params().bias, gen.params().bias + gen.params().theta),
                        max(gen.params().bias, gen.params().bias + gen.params().theta)]
        if param.disc_depth == 1:
            # optimal psi in {-1,1}
            opt_psi = math.copysign(1, np.mean(true_interval) - np.mean(gen_interval))
            mid_of_intervals = 1 / 2 * (np.mean(true_interval) + np.mean(gen_interval))
            # optimal bias : such that disc = 0 at mid_of_intervals
            opt_bias = - opt_psi * mid_of_intervals
            # set optimal parameters for the disc
            with torch.no_grad():
                optimal_disc.disc[0].weight.fill_(opt_psi)
                optimal_disc.disc[0].bias.fill_(opt_bias)
        if param.disc_depth == 2:
            pass

    # TODO other types.

def get_optimal_disc_INTUITION(optimal_disc, gen, param):
    """
    Return the optimal discriminator for given true and fake data.
    :parameter
        - optimal_disc : Discriminator
            optimal discriminator instance
        - trues:
            For param.gan_type == '1D': list[left_end, right_end]
        - fakes:
            For param.gan_type == '1D': list[left_end, right_end]
        - param : AttrDict
            dictionary with additional parameters
    :return: optimal discriminator with updated parameters
    """
    slope_factor = 1
    eps = 0.5

    if param.target_dim == "dirac":
        theta_true = 0  # + param.bias
        theta_gen = gen.params().theta
        # optimal slope : proportional to distance of thetas, at most 1
        opt_psi = slope_factor * (theta_true - theta_gen)
        opt_psi= np.clip(opt_psi, -1, 1)
        with torch.no_grad():
            optimal_disc.disc[0].weight.fill_(opt_psi)

    if param.target_dim == "1D":
        true_interval = [param.data_bias,
                         param.data_bias + param.scale_factor]
        gen_interval = [min(gen.params().bias, gen.params().bias + gen.params().theta),
                        max(gen.params().bias, gen.params().bias + gen.params().theta)]

        if param.disc_depth == 1:
            # optimal slope : proportional to distance of interval means, at most 1
            opt_slope = slope_factor * (np.mean(true_interval) - np.mean(gen_interval))
            opt_slope = np.clip(opt_slope, -1, 1)
            mid_of_intervals = 1 / 2 * (np.mean(true_interval) + np.mean(gen_interval))
            # optimal bias : such that disc = 0 at mid_of_intervals
            opt_bias = - opt_slope * mid_of_intervals
            # set optimal parameters for the disc
            with torch.no_grad():
                optimal_disc.disc[0].weight.fill_(opt_slope)
                optimal_disc.disc[0].bias.fill_(opt_bias)

        if param.disc_depth == 2:
            diff_left = true_interval[0] - gen_interval[0]
            diff_right = true_interval[1] - gen_interval[1]
            mid_point = ((true_interval[0] + gen_interval[0]) / 2 + (true_interval[1] + gen_interval[1]) / 2) / 2


            # # get overlap
            # if (true_interval[0] > gen_interval[1]) or (gen_interval[0] > true_interval[1]):
            #     overlap = None
            # else:
            #     overlap = [np.max([true_interval[0], gen_interval[0]]), np.min([true_interval[1], gen_interval[1]])]
            #
            # # right:
            # # slope positive and proportional to diff_right
            # psi_1_1 = slope_factor * abs(diff_right)
            # psi_1_1 = np.clip(psi_1_1, - 1, 1)
            #
            # # left:
            # # slope negative and proportional to diff_left
            # psi_1_2 = - slope_factor * abs(diff_left)
            # psi_1_2 = np.clip(psi_1_2, - 1, 1)
            #
            # if overlap is None:
            #     act_point = (true_interval[0] + gen_interval[0]) / 2 + (true_interval[1] + gen_interval[1]) / 2
            #     bias_1_1 = - act_point * psi_1_1
            #     bias_1_2 = - act_point * psi_1_2
            #     if diff_right >= 0 :
            #         psi_2_1 = 1
            #     else:
            #         psi_2_1 = - 1
            #     if diff_left >= 0 :
            #         psi_2_2 = - 1
            #     else:
            #         psi_2_2 = 1
            #
            # if overlap is not None:
            #     # right:
            #     if diff_right >= 0 :
            #         # place bias at gen_interval[1] if the intervals not overlap
            #         bias_1_1 = - gen_interval[1] * psi_1_1
            #         # place bias at mid_right if the intervals overlap
            #         # bias_1_1 = - mid_right * psi_1_1
            #         # do not flip the ReLU
            #         psi_2_1 = 1
            #     if diff_right < 0:
            #         # place bias at true_interval[1]
            #         bias_1_1 = - true_interval[1] * psi_1_1
            #         # place bias at mid_right
            #         # bias_1_1 = - mid_right * psi_1_1
            #         # flip the ReLU
            #         psi_2_1 = - 1
            #
            #     # left:
            #     if diff_left >= 0:
            #         # place bias at true_interval[0] if intervals not overlap
            #         bias_1_2 = - true_interval[0] * psi_1_2
            #         # place bias at mid_left if the intervals overlap
            #         # bias_1_2 = - mid_left * psi_1_2
            #         # flip the ReLU
            #         psi_2_2 = - 1
            #     if diff_left < 0:
            #         # place bias at gen_interval[0] if intervals not overlap
            #         bias_1_2 = - gen_interval[0] * psi_1_2
            #         # place bias at mid_left if the intervals overlap
            #         # bias_1_2 = - mid_left * psi_1_2
            #         # do not flip the ReLU
            #         psi_2_2 = 1

            # new idea: same activation points for the ReLUs

            # slope positive and proportional to diff_right
            psi_1_1 = slope_factor * abs(diff_right)
            psi_1_1 = np.clip(psi_1_1, - 1, 1)

            # left:
            # slope negative and proportional to diff_left
            psi_1_2 = - slope_factor * abs(diff_left)
            psi_1_2 = np.clip(psi_1_2, - 1, 1)

            if diff_right >= 0:
                # place bias at gen_interval[1] if the intervals not overlap
                bias_1_1 = - mid_point * psi_1_1
                # place bias at mid_right if the intervals overlap
                # bias_1_1 = - mid_right * psi_1_1
                # do not flip the ReLU
                psi_2_1 = 1
            if diff_right < 0:
                # place bias at true_interval[1]
                bias_1_1 = - mid_point * psi_1_1
                # place bias at mid_right
                # bias_1_1 = - mid_right * psi_1_1
                # flip the ReLU
                psi_2_1 = - 1

            # left:
            if diff_left >= 0:
                # place bias at true_interval[0] if intervals not overlap
                bias_1_2 = - mid_point * psi_1_2
                # place bias at mid_left if the intervals overlap
                # bias_1_2 = - mid_left * psi_1_2
                # flip the ReLU
                psi_2_2 = - 1
            if diff_left < 0:
                # place bias at gen_interval[0] if intervals not overlap
                bias_1_2 = - mid_point * psi_1_2
                # place bias at mid_left if the intervals overlap
                # bias_1_2 = - mid_left * psi_1_2
                # do not flip the ReLU
                psi_2_2 = 1

            with torch.no_grad():
                # "upper" neuron, first layer
                optimal_disc.disc[0].weight[0][0].fill_(psi_1_1)
                optimal_disc.disc[0].bias[0].fill_(bias_1_1)
                # "lower" neuron, first layer
                optimal_disc.disc[0].weight[1][0].fill_(psi_1_2)
                optimal_disc.disc[0].bias[1].fill_(bias_1_2)
                # set both weights in layer 2 to 1/-1, bias to 0
                optimal_disc.disc[2].weight[0][0].fill_(psi_2_1)
                optimal_disc.disc[2].weight[0][1].fill_(psi_2_2)
                optimal_disc.disc[2].bias.fill_(0)

def get_optimal_bias(disc, gen, param):
    if param.target_dim == "1D" and param.disc_depth == 1:
        true_interval = [param.data_bias,
                         param.data_bias + param.scale_factor]
        gen_interval = [gen.params().bias,
                        gen.params().bias + gen.params().theta]
        mid_of_intervals = 1 / 2 * (np.mean(true_interval) + np.mean(gen_interval))
        # optimal bias : such that disc = 0 at mid_of_intervals
        opt_bias = - disc.params().psi * mid_of_intervals
        # set optimal bias for the disc
        with torch.no_grad():
            disc.disc[0].bias.fill_(opt_bias)


# TEST, 1D, disc_depth=2
#
# from model import Discriminator, Generator
# import matplotlib.pyplot as plt
# from attrdict import AttrDict
# # -------------- Specify data, model and training parameters -------------- #
# param = {'device': 'cpu',
#          'target_dim': '1D',  # 'dirac', '1D' or '2D'
#          'scale_factor': 1,
#          'data_bias': -1,
#          'latent_distribution': 'uniform',  # 'dirac', 'uniform', 'gaussian'
#          'z_dim': 1,  # dimension of latent_distribution
#          'gan_type': 'wgan',  # 'wgan' or 'nsgan'
#          'disc_depth': 2,  # depth of discriminator (1 or 2)
#          'n_disc_train': 1,  # disc updates per generator update
#          'lr_disc': 0.01,
#          'lr_gen': 0.01,
#          'n_epochs': 5000,
#          'n_epochs_pic': 25,
#          'batch_size': 25,
#          'pen_weight': 1,  # penalty weight (lambda) for 'wgan'
#          'compare_opt_disc': False  # whether to compare with optimal discriminator
#          }
#
# param = AttrDict(param)
#
# disc = Discriminator(param)
# gen = Generator(param)
#
#
# get_optimal_disc(disc, gen, param)
#
#
# # Discriminator
# x = np.linspace(-3, 3, num=200)
# plt.plot(x, disc(torch.tensor(x).unsqueeze(1).float()).detach().numpy(), color='orange',
#          label="Discriminator", solid_capstyle='round')
# # Target interval
# true_offset = -0.05
# gen_offset = 0.05
# opt_offset = 0.02
# plt.plot([param.data_bias, param.data_bias + param.scale_factor], [true_offset, true_offset],
#      color='blue', lw=5, label="Target Interval", solid_capstyle='round')
# plt.plot([gen.get_params().bias, gen.get_params().bias + gen.get_params().theta],
#                  [gen_offset, gen_offset],
#                  color='red', lw=5, label="Generator", solid_capstyle='round')
# plt.legend()
# plt.show()