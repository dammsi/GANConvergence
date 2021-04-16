# ------------------------ Define Models & Losses ------------------------ #

from attrdict import AttrDict
from torch import nn


class Generator(nn.Module):
    """
    Generator Class
    :parameter:
        - param
    """

    def __init__(self, param):
        super(Generator, self).__init__()  # inherit from nn.Module)
        # Build the "neural network" resembling the generator
        # - uniform input
        # - with bias term, but no activations (affine linear transformation)
        # -> output = theta_1 * input + theta_2
        if param.target_dim == "dirac":
            block = [nn.Linear(in_features=param.z_dim, out_features=1, bias=False)]
        elif param.target_dim == "1D":
            block = [nn.Linear(in_features=param.z_dim, out_features=1, bias=True)]
        elif param.target_dim == "2D":
            block = [nn.Linear(in_features=param.z_dim, out_features=param.n_hidden, bias=True),
                     nn.ReLU()]
            for i in range(1, param.gen_depth):
                if i == param.gen_depth - 1:
                    block.append(nn.Linear(in_features=param.n_hidden, out_features=2, bias=True))
                else:
                    block.append(nn.Linear(in_features=param.n_hidden, out_features=param.n_hidden, bias=True))
                    block.append(nn.LeakyReLU())

        else:
            ValueError("Specify target type correctly to build generator ('dirac','1D' or '2D').")
        # Create generator, named "layer1"
        self.gen = nn.Sequential(*block)

        # set additional parameters
        self.target_dim = param.target_dim

    def forward(self, noise):
        """
        Forward pass of the generator: Map a noise tensor to the target space.
        :parameter:
            - noise: torch.tensor
                a noise tensor with dimensions (n_samples, z_dim)
        """
        return self.gen(noise)

    def params(self):
        """
        Returns the parameters of the generator (as AttrDict).
        """
        if self.target_dim == "dirac":
            theta = self.gen[0].weight[0][0].detach().numpy()
            return AttrDict({'theta': theta})
        elif self.target_dim == "1D":
            theta = self.gen[0].weight[0][0].detach().numpy()
            bias = self.gen[0].bias[0].detach().numpy()
            return AttrDict({'theta': theta,
                             'bias': bias})
        elif self.target_dim == "2D":
            theta_1 = self.gen[0].weight[0][0].detach().numpy()
            theta_2 = self.gen[0].weight[1][0].detach().numpy()
            bias_1 = self.gen[0].bias[0].detach().numpy()
            bias_2 = self.gen[0].bias[1].detach().numpy()
            return AttrDict({'theta_1': theta_1,
                             'theta_2': theta_2,
                             'bias_1': bias_1,
                             'bias_2': bias_2})
        else:
            ValueError("Specify target type correctly to build generator ('dirac','1D' or '2D').")


class Discriminator(nn.Module):
    """
    Discriminator Class
    :parameter:
        - tar_dim : int
            dimension of target data
        - disc_depth : int
            specifies depths of Discriminator, either 1 or 2
        - gan_type : str
            either "wgan" for Wasserstein GAN or "nsgan" for Non-Saturating GAN
    """

    def __init__(self, param):
        super(Discriminator, self).__init__()
        # Build the "neural network" resembling the discriminator
        if param.target_dim == "dirac":
            block = [nn.Linear(in_features=1, out_features=1, bias=False)]
        if param.target_dim == "1D" and param.disc_depth == 1:
            block = [nn.Linear(in_features=1, out_features=1, bias=True)]
        if param.target_dim == "1D" and param.disc_depth == 2:
            block = [nn.Linear(in_features=1, out_features=param.n_hidden, bias=True),
                     nn.ReLU(),  # LeakyReLU, Tanh, Sigmoid...
                     nn.Linear(in_features=param.n_hidden, out_features=1, bias=True)]
        if param.target_dim == "2D" and param.disc_depth == 1:
            block = [nn.Linear(in_features=2, out_features=1, bias=True)]
        if param.target_dim == "2D" and param.disc_depth != 1:
            block = [nn.Linear(in_features=2, out_features=param.n_hidden, bias=True),
                     nn.ReLU()]
            for i in range(1, param.disc_depth):
                if i == param.disc_depth - 1:
                    block.append(nn.Linear(in_features=param.n_hidden, out_features=1, bias=True))
                else:
                    block.append(nn.Linear(in_features=param.n_hidden, out_features=param.n_hidden, bias=True))
                    block.append(nn.ReLU())
        # finalize by Sigmoid layer for nsgan
        if param.gan_type == "nsgan" or param.gan_type == "vanilla":
            # Sigmoid is also used in original dirac GAN! (Without sigmoid it's not working...)
            block.append(nn.Sigmoid())
        # create the discriminator
        self.disc = nn.Sequential(*block)

        # set additional parameters
        self.target_dim = param.target_dim
        if 'disc_depth' in param:
            self.disc_depth = param.disc_depth

    def forward(self, sample):
        """
        Forward pass of the discriminator: Map from target space to decision space,
        i.e, the real line for gan_type=="wgan" (slope), or [0,1] for "nsgan".
        :parameter:
            - sample: torch.tensor
                a sample tensor with dimensions (n_samples, tar_dim)
        """
        return self.disc(sample)

    def params(self):
        """
        Returns the parameters of the discriminator (as np.array).
        """
        if self.target_dim == "dirac":
            psi = self.disc[0].weight[0][0].detach().numpy()
            return AttrDict({'psi': psi})

        elif self.target_dim == "1D" and self.disc_depth == 1:
            psi = self.disc[0].weight[0][0].detach().numpy()
            bias = self.disc[0].bias[0].detach().numpy()
            return AttrDict({'psi': psi,
                             'bias': bias})
        elif self.target_dim == "1D" and self.disc_depth == 2:
            # layer 1
            psi_1_1 = self.disc[0].weight[0][0].detach().numpy()
            psi_1_2 = self.disc[0].weight[1][0].detach().numpy()
            bias_1_1 = self.disc[0].bias[0].detach().numpy()
            bias_1_2 = self.disc[0].bias[1].detach().numpy()
            # layer 2
            psi_2_1 = self.disc[2].weight[0][0].detach().numpy()
            psi_2_2 = self.disc[2].weight[0][1].detach().numpy()
            bias_2 = self.disc[2].bias[0].detach().numpy()
            return AttrDict({'psi_1_1': psi_1_1,
                             'psi_1_2': psi_1_2,
                             'bias_1_1': bias_1_1,
                             'bias_1_2': bias_1_2,
                             'psi_2_1': psi_2_1,
                             'psi_2_2': psi_2_2,
                             'bias_2': bias_2})

        elif self.target_dim == "2D" and self.disc_depth == 1:
            psi_1 = self.disc[0].weight[0][0].detach().numpy()
            psi_2 = self.disc[0].weight[0][1].detach().numpy()
            bias = self.disc[0].bias[0].detach().numpy()
            return AttrDict({'psi_1': psi_1,
                             'psi_2': psi_2,
                             'bias': bias})
        # macht das hier eigentlich Sinn?
        elif self.target_dim == "2D" and self.disc_depth == 2:
            # layer 1
            psi_1_11 = self.disc[0].weight[0][0].detach().numpy()
            psi_1_12 = self.disc[0].weight[0][1].detach().numpy()
            psi_1_21 = self.disc[0].weight[1][0].detach().numpy()
            psi_1_22 = self.disc[0].weight[1][1].detach().numpy()
            bias_1_1 = self.disc[0].bias[0].detach().numpy()
            bias_1_2 = self.disc[0].bias[1].detach().numpy()
            # layer 2
            psi_2_1 = self.disc[2].weight[0][0].detach().numpy()
            psi_2_2 = self.disc[2].weight[0][1].detach().numpy()
            bias_2 = self.disc[2].bias[0].detach().numpy()
            return AttrDict({'psi_1_11': psi_1_11,
                             'psi_1_12': psi_1_12,
                             'psi_1_21': psi_1_21,
                             'psi_1_22': psi_1_22,
                             'bias_1_1': bias_1_1,
                             'bias_1_2': bias_1_2,
                             'psi_2_1': psi_2_1,
                             'psi_2_2': psi_2_2,
                             'bias_2': bias_2})
        else:
            ValueError("Specify target type correctly ('dirac','1D' or '2D').")
