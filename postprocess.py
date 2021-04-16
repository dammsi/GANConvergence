# Postprocess
import numpy as np
import torch
from matplotlib import pyplot as plt
import yaml

from data import latent_data
from data import target_data


def saving(param, disc, gen):
    """
    Save parameter, models and/or last iterate to save_directory specified in param.
    :param param: AttrDict with all specifications of the training & models.
    :param disc: Discriminator instance.
    :param gen: Generator instance.
    """
    save_the_models = False
    save_the_params = True
    save_last_figure = False

    # ------------ Last Figure ------------ #
    if save_last_figure:
        # Image Generation
        true_samples = target_data(param, n_samples=500)
        gen_samples = gen(latent_data(param, n_samples=500)).detach().numpy()

        # Discriminator = color grid
        x_span = np.linspace(-3, 3)
        y_span = np.linspace(-3, 3)
        xx, yy = np.meshgrid(x_span, y_span)
        grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])

        z = disc(grid).view(xx.shape).detach().numpy()
        # Target samples
        plt.plot(true_samples[:, 0], true_samples[:, 1], marker='o', color='blue', alpha=0.4,
                 markersize=5, linestyle='None', label='Target Samples')
        # Generator samples
        plt.plot(gen_samples[:, 0], gen_samples[:, 1], marker='o', color='red', alpha=0.4,
                 markersize=5, linestyle='None', label='Generator Samples')
        plt.legend()

        plt.savefig(f"{param.save_dir}/FinalState.png", dpi=150)
        # plt.show()
        plt.clf()

    # ------------- Parameters ------------- #
    if save_the_params:
        # Save the parameters
        with open(f'{param.save_dir}/param.yml', 'w') as outfile:
            yaml.dump(param, outfile, default_flow_style=False)

    # To Load the parameters:
    # with open(f'{param.save_dir}/param.yml', 'r') as stream:
    #    params_loaded = yaml.full_load(stream)

    # --------------- Model --------------- #
    if save_the_models:
        # Saving the models
        torch.save(disc.state_dict(), f"{param.save_dir}/disc.pt")
        torch.save(gen.state_dict(), f"{param.save_dir}/gen.pt")

# count number of (trainable) parameters
# sum(p.numel() for p in model.parameters() if p.requires_grad)
