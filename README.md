# GAN Convergence

Code for the experiments in "Convergence Properties of Generative Adversarial Networks".

We implemented the small-scale experiments in PyTorch to explore different techniques fostering GAN convergence. 
The code base covers different GAN objectives including non-saturating GAN, WGAN, WGAN-GP, WGAN-LP and WGAN with a simple gradient regularization. 
The learning scenarios cover the DiracGAN, learning intervalls in 1D and different manifolds in 2D with respective visualizations.

## Usage
For each experiment (Dirac: `<DiracGAN.py>`, 1D: `<GAN_1D.py>`, 2D: `<GAN_2D.py>`) simply specified the required parameters and select whether you want singleruns, multiruns or experiments (multiruns on all combinations of two parameter sets).
