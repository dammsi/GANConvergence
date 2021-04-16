# ----------------- Visualize the current learning progress ----------------- #

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import torch
from matplotlib import cm


def visualize_training(gen, disc, epoch, true_samples, gen_samples, param,
                       disc_losses, gen_losses, W_dist, trajectory_points=None,
                       decision_boundaries=None):
    """
    Visualize training and save figures.
    The plot consists of two minor plots (Losses, Wasserstein Distance or Trajectories)
    and one larger plot depicting the current learning progress (in 2D or 3D).
    :param gen: Generator instance
    :param disc: Discriminator instance
    :param epoch: current epoch (int)
    :param true_samples: samples from target distribution
    :param gen_samples: samples from the current generator instance
    :param param: AttrDict with additional parameters
    :param disc_losses: Losses for current Discriminator
    :param gen_losses: Losses for current Generator
    :param W_dist: List of approximated Wasserstein distances
    :param trajectory_points: Trajectories (optional)
    :param decision_boundaries: Decision boundaries (optional)
    """

    plt.style.use('ggplot')
    fig = plt.figure(1, (14, 7))
    gridspec.GridSpec(2, 3)  # 2 rows, 3 columns

    # --- SUBPLOT 1: Losses / Wasserstein Distance --- #
    plt.subplot2grid((2, 3), (0, 0), colspan=1, rowspan=1)
    # plt.title('Losses')
    # plt.xlabel("epoch")
    plt.xlim(0, param.n_epochs)
    # Choose whether Losses or Wasserstein Distance (or both) should be plotted
    if param.target_dim != "dirac":
        plt.plot([param.n_epochs_loss * i for i in range(len(gen_losses))], gen_losses,
                 color='red')  # label="Generator Loss")
        plt.plot([param.n_epochs_loss * i for i in range(len(disc_losses))], disc_losses,
                 color='orange')  # label="Discriminator Loss")
    else:
        plt.ylim(bottom=0., top=max(W_dist))
        # plt.ylabel(r"$\widehat{W}_1(\mathbb{P},\mathbb{Q})$")
        plt.ylabel(r"$\widehat{W}_1$")
        plt.xlabel("epochs")
        plt.plot([param.n_epochs_loss * i for i in range(len(W_dist))], W_dist, color='blue')

    # --- SUBPLOT 2: Trajectory plot (if possible, else Wasserstein Distance) --- #
    trajectory_subfig = plt.subplot2grid((2, 3), (1, 0), colspan=1, rowspan=1)

    if param.target_dim == "dirac":
        plot_trajectories_dirac(trajectory_subfig, trajectory_points, param)
    elif param.target_dim == "1D" and param.disc_depth == 1:
        plot_trajectroies_1D(trajectory_subfig, trajectory_points)
    else:
        plt.xlim(0, param.n_epochs)
        plt.ylim(bottom=0., top=max(W_dist))
        # plt.ylabel(r"$\widehat{W}_1(P,Q)$")
        plt.ylabel(r"$\widehat{W}_1$")
        plt.xlabel("epochs")
        plt.plot([param.n_epochs_loss * i for i in range(len(W_dist))], W_dist, color='blue')

    # --- SUBPLOT 3: Current state of training --- #
    visualize3d = False  # Choose whether visualization should be 2D or 3D
    visualize3d = False  # Choose whether visualization should be 2D or 3D
    if visualize3d:
        subplot = plt.subplot2grid((2, 3), (0, 1), colspan=2, rowspan=2, projection='3d')
    else:
        subplot = plt.subplot2grid((2, 3), (0, 1), colspan=2, rowspan=2)

    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    visualize_current_state(subplot, disc, gen, true_samples, gen_samples,
                            decision_boundaries, param, visualize3d=visualize3d)

    # Set title and file name
    # plt.title(f"{param.gan_type} - Epoch {epoch}")
    plt.legend(loc=2)
    if epoch == param.n_epochs:
        # tikzplotlib.clean_figure()
        tikzplotlib.save(f"{param.save_dir}/epoch_{epoch}.tex")
    plt.savefig(f"{param.save_dir}/epoch_{epoch}.png", dpi=100)
    # plt.show()
    plt.clf()


def plot_trajectories_dirac(ax, trajectory_points, param):
    """
    Subroutine to plot trajectories for the Dirac GAN on given axis.
    :param ax: axis for plotting
    :param trajectory_points: Trajectory points to plot.
    :param param : AttrDict instance with data_bias
    """
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\psi$")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    plt.plot(trajectory_points[0, 0], trajectory_points[0, 1], marker='d', markersize=5, color='red')
    plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], marker='.', markersize=5, color='red', alpha=0.4)
    plt.plot(param.data_bias, 0, marker='*', color='blue', alpha=0.8)  # target
    return ax


def plot_trajectroies_1D(ax, trajectory_points):
    """
    Subroutine to plot trajectories for 1D GANs on given axis.
    :param ax: axis for plotting
    :param trajectory_points: Trajectory points to plot.
    """
    plt.xlabel(r"$\theta, b_G$")
    plt.ylabel(r"$\psi, b_D$")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    plt.plot(trajectory_points[0, 0], trajectory_points[0, 1], marker='d', markersize=5, color='red')
    plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], marker='.', markersize=5, color='red', alpha=0.4,
             label=r"$\theta$ vs. $\psi$")

    plt.plot(trajectory_points[0, 2], trajectory_points[0, 3], marker='d', markersize=5, color='chocolate')
    plt.plot(trajectory_points[:, 2], trajectory_points[:, 3], marker='.', markersize=5, color='chocolate', alpha=0.4,
             label=r"$b_G$ vs. $b_D$")
    plt.legend(loc=4)
    return ax


def visualize_current_state(ax, disc, gen, true_samples, gen_samples, decision_boundaries, param,
                            visualize3d=False):
    """
    Subroutine to visualize the current state of the GAN game.
    :param ax: axis for plotting
    :param disc: Discriminator instance
    :param gen: Generator instance
    :param true_samples: samples from target_distribution()
    :param noise: samples from latent_distribution()
    :param decision_boundaries: Decision boundaries of the Discriminator instance
    :param param: AttrDict instance with additional parameter, e.g., target_dim, gan_type
    :param visualize3d: Bool specifying whether to plot in 2D or 3D.
    """
    true_offset = -0.05
    gen_offset = 0.05

    if param.target_dim == "dirac":
        true_samples = true_samples[-1]
        gen_samples = gen_samples.detach()[-1]
        x = np.linspace(-5, 5)
        # linear discriminator
        plt.plot(x, disc(torch.tensor(x).unsqueeze(1).float()).detach().numpy(), color='orange', label="Discriminator")
        if param.gan_type == "wgan":  # plot lipschitz boundaries
            plt.plot(x, 1 * x, color='grey', alpha=0.2)
            plt.plot(x, - 1 * x, color='grey', alpha=0.2)
        # Target
        plt.plot(true_samples, 0, marker='o', color='blue',
                 markersize=12, linestyle='None', label="Target", alpha=0.8)
        # Generator
        plt.plot(gen_samples, 0, marker='o', color='red',
                 markersize=10, linestyle='None', label='Generator', alpha=0.8)

    if param.target_dim == "1D":
        # Discriminator
        x = np.linspace(-5, 5)
        D_offset = disc(torch.zeros(1).float()).detach().numpy()
        plt.plot(x, disc(torch.tensor(x).unsqueeze(1).float()).detach().numpy() - D_offset, color='orange',
                 label="Discriminator", solid_capstyle='round')
        # Discriminator - Lipschitz boundaries for disc_depth==1
        if param.gan_type == "wgan" and param.disc_depth == 1:
            plt.plot(x, 1 * x,  # + disc.params().bias / disc.params().psi,
                     color='grey', alpha=0.2)
            plt.plot(x, - 1 * x,  # - disc.params().bias / disc.params().psi,
                     color='grey', alpha=0.2)
        # Target interval
        plt.plot([param.data_bias, param.data_bias + param.scale_factor], [true_offset, true_offset],
                 color='blue', lw=5, label="Target Interval", solid_capstyle='round')
        # Target samples
        plt.plot(true_samples, np.zeros_like(true_samples) + 2 * true_offset, marker='o', color='blue', alpha=0.4,
                 markersize=5, linestyle='None')  # label='Target Samples')
        # Generator interval
        plt.plot([gen.params().bias, gen.params().bias + gen.params().theta],
                 [gen_offset, gen_offset],
                 color='red', lw=5, label="Generator", solid_capstyle='round')
        # Generator samples
        gen_samples = gen_samples.detach().numpy()
        plt.plot(gen_samples, np.zeros_like(gen_samples) + 2 * gen_offset, marker='o', color='red', alpha=0.4,
                 markersize=5, linestyle='None')  # label='Generator Samples')
        # if param.disc_depth == 2 and decision_boundaries is not None:
        # plot decision boundaries
        # plt.vlines(decision_boundaries[-1], -3, 3, color='gray', alpha=0.2, label="Decision Boundaries")

    if param.target_dim == "2D":
        # Discriminator as a color grid
        x_span = np.linspace(-3, 3)
        y_span = np.linspace(-3, 3)
        xx, yy = np.meshgrid(x_span, y_span)
        grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])

        z = disc(grid).view(xx.shape).detach().numpy()

        if visualize3d:
            ax.plot_surface(xx, yy, z, alpha=0.5)
            cset = ax.contourf(xx, yy, z, zdir='z', offset=-2, cmap=cm.coolwarm, alpha=0.2)
            # cset = ax.contourf(xx, yy, z, zdir='x', offset=-4, cmap=cm.coolwarm, alpha=0.2)
            # cset = ax.contourf(xx, yy, z, zdir='y', offset=3, cmap=cm.coolwarm, alpha=0.2)

            ax.set_xlabel('X')
            ax.set_xlim(-4, 4)
            ax.set_ylabel('Y')
            ax.set_ylim(-3, 3)
            ax.set_zlabel('Z')
            ax.set_zlim(-2, 2)
        else:
            cs = plt.pcolormesh(xx, yy, z, shading='auto', alpha=0.2)
            plt.colorbar(cs, shrink=0.9)

            # plot decision boundaries of D
            # x = np.linspace(-4, 4)
            # for i in range(param.disc_depth - 1):
            #     d_weights = disc.disc[2 * i].weight.detach().tolist()
            #     d_bias = disc.disc[2 * i].bias.detach().tolist()
            #     for j in range(param.n_hidden):  # iterate over hidden neurons
            #         y = -1 / d_weights[j][0] * (d_weights[j][1] * x + d_bias[j])
            #         alpha = 0.5 - i / 5
            #         plt.plot(x, y, color="gray", alpha=alpha)

        # Target samples
        plt.plot(true_samples[:, 0], true_samples[:, 1], marker='o', color='blue', alpha=0.4,
                 markersize=5, linestyle='None', label='Target Samples')
        # Generator samples
        gen_samples = gen_samples.detach().numpy()
        plt.plot(gen_samples[:, 0], gen_samples[:, 1], marker='o', color='red', alpha=0.4,
                 markersize=5, linestyle='None', label='Generator Samples')
        plt.legend()
    return ax


def plot_multiple_runs(W_losses, param, name_1=None, name_2=None, current_value_1=None, current_value_2=None):
    """
    Subroutine to visualize progress over multiple runs.
    :param W_losses: List with approximated Wasserstein losses.
    :param param: AttrDict with additional parameters
    :param name_1: Name of the first parameter (in experiment)
    :param name_2: Name of the second parameter (in experiment)
    :param current_value_1: Value of the first parameter (in experiment)
    :param current_value_2: Value of the second parameter (in experiment)
    :return:
    """

    plt.style.use('ggplot')
    # prepare array for sum & different colors
    W_losses_sum = np.zeros_like(W_losses[next(iter(W_losses))])
    color = iter(cm.Blues(np.linspace(0.33, 0.8, len(W_losses))))
    # plot W_loss for each run
    for run in W_losses.keys():
        c = next(color)
        plt.plot([param.n_epochs_loss * i for i in range(len(W_losses[run]))], W_losses[run], color=c, alpha=0.4)
        W_losses_sum += np.array(W_losses[run])
    # plot average W_loss
    plt.plot([param.n_epochs_loss * i for i in range(len(W_losses[run]))], W_losses_sum / len(W_losses), color='blue')
    # plt.ylabel(r"$\widehat{W}_1(\mathbb{P},\mathbb{Q})$")
    plt.ylabel(r"$W_1$")
    plt.xlabel("epochs")
    if name_1 is not None:
        tikzplotlib.save(f"{param.save_dir}/W_{name_1}{current_value_1}_{name_2}{current_value_2}.tex")
        plt.savefig(f"{param.save_dir}/W_{name_1}{current_value_1}_{name_2}{current_value_2}.png", dpi=100)
    else:
        tikzplotlib.save(f"{param.save_dir}/W_losses.tex")
        plt.savefig(f"{param.save_dir}/W_losses.png", dpi=100)
    plt.clf()
