# Projected GAN with gradient penalty training

This is a version of Projected GAN where the discriminator is trained with a WGAN-GP-type objective, using a one-sided gradient penalty to approximate a Wasserstein-1 potential. It was made as part of a research project at the University of Toronto. The full Projected GAN discriminator consists of a feature network that outputs features at several depths, followed by a "mini-discriminator" that tries to distinguish between features from real images and features from generated images at each depth. There are several ways to implement a gradient penalty for the Projected GAN discriminator; two are available here. The gp_source argument in train.py governs which gradient penalty implementation is used.

* When gp_source == 'features', a gradient penalty is applied separately to each of the mini-discriminators, without including the feature network. This means that for each mini-discriminator, the gradient of the output value with respect to the input features is penalized.

* When gp_source == 'images', a single gradient penalty is applied to the whole discriminator including the feature network. This means that the gradient of the output of the full discriminator (i.e. the sum of the outputs of all the mini-dscriminators) with respect to the image inputs of the feature network is penalized. 

The gp_source == 'images' gradient penalty requires backpropagating through the feature network requires backpropagating through the feature network, which considerably slows down the code. On the other hand, the gp_source == 'fearures' options ignores the feature network's gradient. 


### What has been changed in this forked repo?

Compared to the original Projected GAN repository, the following changes have been made:

* train.py now takes new arguments pertaining to the WGAN-GP loss function. These new arguments are:
  * 'd_loss': Decides whether the new WGAN-GP loss or the original Projected GAN loss is used to train the discriminator. Can be either 'WGAN_GP_Loss' or 'ProjectedGANLoss'
  * 'gp_source': Explained above. Can be either 'features' or 'images'.
  * 'gp_lambda': Weight given to the gradient penalty in the full dircriminator loss function. (Float>0)
  * 'gp_clamp': Size of the gradient's norm under which no penalty is applied (in a standard WGAN-GP objective, this value is 1 by default). (Float>=0)
* training/training_loop.py now adds a forward hook on the feature network, giving convenient access to its inputs and outputs for gradient computations.
* training/loss.py now includes the 'WGAN_GP_Loss' loss function.
* I've replaced the environment.yml with a requirements.txt file to be installed with Python 3.8 or above. 


### Experimental results
Coming soon.

