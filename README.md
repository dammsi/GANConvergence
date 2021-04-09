# GAN Convergence

Code for the experiments in "Convergence Properties of Generative Adversarial Networks".

We implemented the small-scale experiments in PyTorch (v1.8.1) to explore different techniques fostering GAN convergence. 
The code base covers different GAN objectives including non-saturating GAN, WGAN, WGAN-GP, WGAN-LP and WGAN with a simple gradient regularization.
The learning scenarios cover the DiracGAN, learning intervalls in 1D, different manifolds in 2D and mixture of Gaussians in 2D with respective visualizations.

## Usage
Specify the required parameters for each experiment (Dirac: `DiracGAN.py`, 1D: `GAN_1D.py`, 2D: `GAN_2D.py`) and select whether you want singleruns, multiruns or experiments (multiruns on all combinations of two parameter lists).
