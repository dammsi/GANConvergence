# ----- VISUALIZE ----- #

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import tikzplotlib


def visualize_training(gen, disc, epoch, true_samples, gen_samples, param,
                       disc_losses, gen_losses, W_dist, trajectory_points=None,
                       decision_boundaries=None):
    """ todo
    Visualize training and save figures.
    :parameter:
        - gen
            Generator instance.
        - disc
            Discriminator instance.
        - epoch : int
            Current epoch number
        - true_samples : torch.tensor
            Samples from data.target_data()
        - noise : torch.tensor
            Data from data.latent_data()
        - param : AttrDict
            Dictionary containing additional parameters
        - disc_losses & gen_losses : Agents individual losses
        - W_dist : Wasserstein distance
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
        plt.plot([param.n_epochs_loss * i for i in range(len(gen_losses))], gen_losses, color='red')  # label="Generator Loss")
        plt.plot([param.n_epochs_loss * i for i in range(len(disc_losses))], disc_losses, color='orange')  # label="Discriminator Loss")
    else:
        plt.ylim(bottom=0., top=max(W_dist))
        #plt.ylabel(r"$\widehat{W}_1(\mathbb{P},\mathbb{Q})$")
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
        #plt.ylabel(r"$\widehat{W}_1(P,Q)$")
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
        #tikzplotlib.clean_figure()
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
    """ todo
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
    opt_offset = 0.02

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
        plt.plot(x, disc(torch.tensor(x).unsqueeze(1).float()).detach().numpy(), color='orange',
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
        #if param.disc_depth == 2 and decision_boundaries is not None:
            # plot decision boundaries
            #plt.vlines(decision_boundaries[-1], -3, 3, color='gray', alpha=0.2, label="Decision Boundaries")

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

def plot_multiple_runs(W_losses, param, name_1 = None, name_2 = None, current_value_1 = None, current_value_2 = None):
    """
    Plot
    :param W_losses: Dictionary with approximate Wasserstein losses for each run.
    :param param: AttrDict instance.
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
    #plt.ylabel(r"$\widehat{W}_1(\mathbb{P},\mathbb{Q})$")
    plt.xlabel("epochs")
    if name_1 is not None:
        tikzplotlib.save(f"{param.save_dir}/W_{name_1}{current_value_1}_{name_2}{current_value_2}.tex")
        plt.savefig(f"{param.save_dir}/W_{name_1}{current_value_1}_{name_2}{current_value_2}.png", dpi=100)
    else:
        tikzplotlib.save(f"{param.save_dir}/W_losses.tex")
        plt.savefig(f"{param.save_dir}/W_losses.png", dpi=100)
    plt.clf()

def draw_neural_net(ax, layer_sizes, layer_title=None, weights=None, show_weights=False, with_biases=True):
    """
    Visualize a neural net using matplotilb.
    Currently supported: Fullyconected neural nets with or without biases.
    Now: Weight given as tuples: [[1st layer], [bias], [second layer], [bias] ...]
    :parameter:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    :usage:
        fig = plt.figure(figsize=(12, 12))
        draw_neural_net(fig.gca(), [4, 7, 2])
    """
    # for subplot copy these lines out and give "ax" as an argument
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.gca()
    ax.axis('off')

    left = 0.15
    right = 0.85
    bottom = 0.1
    top = 0.9
    # THIS Code checks wether given weights from a Pytorch model fits to layer_sizes
    # if weights != None:
    #     shapes = []
    #     for i in weights:
    #         shapes.append(np.array(i.size()))
    #     # check the weight dimensions:
    #     if with_biases==False:
    #         for i, shape in enumerate(shapes):
    #             if layer_sizes[i] != shape[1]:
    #                 # shape = [out, in]
    #                 raise Exception("'layer_sizes' and weight matrix do not match. Have you checked for the biases?.")
    #     else:
    #         for i, shape in enumerate(shapes):
    #             if i%2==0 and layer_sizes[i//2] != shape[1]:
    #                 # only check every second entry in shape corresponding to weight matrices
    #                 raise Exception("'layer_sizes' and weight matrix do not match. Have you checked for the biases?.")
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)
    neuron_frac = v_spacing / 4.
    if layer_sizes == [1, 1]:
        neuron_frac = v_spacing / 9.

    # Adjust font size:
    if neuron_frac >= 0.1:
        fontsize = 15
    elif neuron_frac > 0.05:
        fontsize = 12
    else:
        fontsize = 8

    # Nodes & Bias
    bias_pos = []
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            x = n * h_spacing + left
            y = layer_top - m * v_spacing
            circle = plt.Circle((x, y), neuron_frac,
                                color='w', ec='k', zorder=4)
            if with_biases and n > 0 and m == layer_size - 1:
                x_bias = x - h_spacing / 2.  # half the way to previous layer
                if layer_sizes[n - 1] > layer_sizes[n]:
                    y_bias = y - (layer_sizes[n - 1] - layer_sizes[
                        n]) * v_spacing / 2  # move bias down accordingly level of neuron
                elif layer_sizes[n - 1] == layer_sizes[n]:
                    y_bias = y - v_spacing / 2
                else:
                    y_bias = y  # bias at level of neuron
                bias = plt.Circle((x_bias, y_bias), neuron_frac / 2.,
                                  color='w', ec='k', zorder=3)
                bias_pos.append([x_bias, y_bias])  # store for edge drawing
                # set 1 for each bias
                # ax.annotate("1", xy=(x_bias, y_bias), ha="center", va="center", zorder=5, fontsize=fontsize)
                ax.add_artist(bias)
            # annotate neuron names
            # ax.annotate("{}_{}".format(n,m), xy=(x, y), ha="center", va="center", zorder=5, fontsize=fontsize)
            ax.add_artist(circle)
        # Layer names
        if layer_title is not None:
            plt.text(
                x=n * h_spacing + left,
                y=top + 0.05,
                s="{}".format(layer_title[n]),
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=fontsize,
                color='black'
            )

    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                frac = 5  # fraction to interpolate along edge, 2 = middle, higher number = closer to neuron
                frac_bias = 6
                # Node to Node
                x1_line = n * h_spacing + left
                x2_line = (n + 1) * h_spacing + left
                y1_line = layer_top_a - m * v_spacing
                y2_line = layer_top_b - o * v_spacing
                line = plt.Line2D([x1_line, x2_line],
                                  [y1_line, y2_line], c='k', alpha=0.5)
                if show_weights:
                    ax.annotate("{}".format(weights[0][0]),  # FIXME
                                xy=(x1_line / frac + x2_line * (frac - 1) / frac,
                                    y1_line / frac + y2_line * (frac - 1) / frac),
                                ha="center", va="bottom", zorder=5, fontsize=fontsize - 2)
                ax.add_artist(line)
                # Bias to Node
                if m == 0 and with_biases:  # just needed once per "m"
                    line = plt.Line2D([bias_pos[n][0], (n + 1) * h_spacing + left],
                                      [bias_pos[n][1], layer_top_b - o * v_spacing], c='k', alpha=0.5, linestyle=':')
                    if show_weights:
                        ax.annotate("{}".format(weights[1][0]),  # FIXME
                                    xy=(bias_pos[n][0] / frac_bias + ((n + 1) * h_spacing + left) * (
                                            frac_bias - 1) / frac_bias,
                                        bias_pos[n][1] / frac_bias + (layer_top_b - o * v_spacing) * (
                                                frac_bias - 1) / frac_bias - v_spacing / 12),
                                    ha="right", va="bottom", zorder=6, fontsize=fontsize - 2)
                    ax.add_artist(line)
    return ax
