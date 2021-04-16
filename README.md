# GAN Convergence

Code for the experiments in "Convergence Properties of Generative Adversarial Networks".

We implemented the small-scale experiments in PyTorch (v1.8.1) to explore different techniques fostering GAN convergence. 
The code base covers different GAN objectives including non-saturating GAN, WGAN, WGAN-GP, WGAN-LP and WGAN with a simple gradient regularization.
The learning scenarios cover the DiracGAN, learning intervalls in 1D, different manifolds in 2D and mixture of Gaussians in 2D with respective visualizations.

## Usage
Specify the required parameters for each experiment (Dirac: `DiracGAN.py`, 1D: `GAN_1D.py`, 2D: `GAN_2D.py`) and select whether you want singleruns, multiruns or experiments (multiruns on all combinations of two parameter lists).

### Dirac GAN
Visualization of training a WGAN + simple Gradient Penalty in the DiracGAN setting.
<img src="./images/DiracGAN.gif" width="100%" height="100%">
**Upper left:** Wasserstein Distance (blue) between target distribution (Dirac at <img src="https://latex.codecogs.com/svg.latex?\space0" title="P_Circle" />) and generator distribution (Dirac at <img src="https://latex.codecogs.com/svg.latex?\space\theta" title="P_Circle" />)
**Lower left:** Trajectories of Generator (<img src="https://latex.codecogs.com/svg.latex?\space\theta" title="P_Circle" />) and Discriminator (<img src="https://latex.codecogs.com/svg.latex?\space\psi" title="P_Circle" />) 
**Right:** Depiction of current learning progress

### 1D: Interval Learning
Learning of a uniform distribution with a shallow neural net as discriminator. Successful learning (simple gradient penalty) and failed learning (WGAN-GP).
<p float="left">
  <img src="./images/1D_sGP.gif" width="49%" height="50%">
  <img src="./images/1D_WGAN-GP.gif" width="49%" height="50%">
</p>


### 2D: Manifold Learning

With `GAN_2D.py` learning of different manifolds  or a mixture of Gaussians
<p float="left">
  <img src="./images/2D_square.gif" width="33%" height="33%">
  <img src="./images/2D_circle.gif" width="33%" height="33%">
  <img src="./images/2D_5gauss.gif" width="33%" height="33%">
</p>
(Click to enlarge)
