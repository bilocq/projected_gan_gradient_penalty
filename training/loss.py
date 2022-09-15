# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Modified by Axel Sauer for "Projected GANs Converge Faster".
#
# Modified again by Étienne Bilocq for PG-GP.

import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F
from torch_utils import training_stats
from torch_utils.ops import upfirdn2d
from torch import autograd


class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()



#############################
### WGAN-GP loss function ###
#############################

# This loss function trains the Projected GAN multiscaled feature space discriminator with a WGAN-GP type loss function.
# A one-sided gradient penalty is applied separately to each of the mini-discriminator that make up the full multiscaled
# discriminator.
# 
# OPTIONS
#     gp_source: (str) Decides whether to include the feature network in the gradient penalty computation. Can be
#                'features' or 'images. 
#                - If gp_source == 'features', then a gradient penalty is computed separately for each of the feature-
#                  space mini-discriminators with respect to their corresponding feature-space inputs (here, feature-space
#                  refers to the output space of the feature network). This means that backpropagating the gradient 
#                  penalty only requires backpropagating through the mini-discriminators, not through the feature network.
#                  Training Projected GAN with WGAN_GP_Loss and gp_source == 'features' is roughly as fast as training it 
#                  with the original loss function ProjectedGANLoss.
#                - If gp_source == 'images', then the gradient being penalized is that of the whole PG discriminator 
#                  consisting in the composition of the multiscaled feature-space discriminator with the feature network.
#                  This requires bckpropagating through the feature network, considerably slowing down the algorithm.
#     gp_lambda: Weight given to the gradient penalty in the whole discriminato loss function.
#     gp_clamp: Size of the gradient's norm under which no gradient penalty is applied. A standard WGAN-GP loss function
#               has gp_clamp == 1, but we've obtained improvements modifying this value.

class WGAN_GP_Loss(Loss):
    def __init__(self, device, G, D, G_ema, blur_init_sigma=0, blur_fade_kimg=0, gp_source='features', gp_lambda=10, gp_clamp=0.5, **kwargs):
        super().__init__()
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        
        # Gradient penalty options
        self.gp_source = gp_source
        self.gp_lambda = gp_lambda
        self.gp_clamp = gp_clamp

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        img = self.G.synthesis(ws, c, update_emas=False)
        return img

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())

        logits = self.D(img, c)
        return logits


    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg', 'Greg']: return  # no regularization needed for PG

        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0


        if do_Gmain:
        
            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img = self.run_G(gen_z, gen_c)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                loss_Gmain = (-gen_logits).mean()

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()
          
        if do_Dmain:
            # WGAN-GP loss function with gradient penalty on sampled along straight lines in feature space.

            with torch.autograd.profiler.record_function('D_GP_divergence'):
                
                ### Get real and fake images
                gen_img = self.run_G(gen_z, gen_c, update_emas=True)
                real_img_tmp = real_img.detach().requires_grad_(False)
                
                ### Minimize for fakes images and maximize for real images
                Dval_g = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma).mean() # g for generated
                images_g = self.D.feature_networ.input # images_g not necessarily equal to gen_img when using differentiable augmentation
                features_g = self.D.feature_network.output # Keep to get interpolated features for gradient penalty with straight line sampling
                Dval_g.backward()
                Dval_r_minus = -self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma).mean() # r for real
                images_r = self.D.feature_network.input
                features_r = self.D.feature_network.output # Keep to get interpolated features for gradient penalty with straight line sampling
                Dval_r_minus.backward()
                training_stats.report('Loss/D/Divergence', -Dval_r_minus - Dval_g)
                
            with torch.autograd.profiler.record_function('D_GP_Grad_Penalty'):
                
                ### Gradient penalty
                # Draw interpolation values
                dist = stats.uniform
                t_inter = torch.from_numpy(dist.rvs(size=(gen_img.shape[0],1,1,1)).astype(np.float32)).to(self.device)
               
                if self.gp_source == 'features':                    
                    # Compute points in feature space where gradient penalty will be computed and apply corresponding mini-discs
                    # to these points.
                    gp_points = [] # Points in feature space where gradient penalty is applied
                    Dvals_gp = []
                    for key, disc in self.D.discriminator.mini_discs.items():
                        interpolate = (1-t_inter)*features_g[key] + t_inter*features_r[key]
                        interpolate.requires_grad = True
                        gp_points.append(interpolate)
                        Dvals_gp.append(disc(interpolate, None).view(interpolate.size(0), -1))
                    Dvals_gp = torch.cat(Dvals_gp, dim=1) 
                    
                    # Compute gradient penalty
                    gradients = autograd.grad(outputs      = Dvals_gp, 
                                              inputs       = gp_points,
                                              grad_outputs = torch.ones_like(Dvals_gp),
                                              create_graph = True)
                    grad_pen = 0
                    for gradi in gradients:
                        gradi = gradi.reshape(gradi.size(0), -1)
                        grad_pen += (torch.clamp(gradi.norm(2, dim=1) - self.gp_clamp, min=0)**2).mean() * self.gp_lambda
                
                elif self.gp_source == 'images':
                    # Compute points in image space where gradient penalty will be computed and apply D (i.e. feature_network
                    # followed by multiscaled discriminator) to these points.
                    interpolate = (1-t_inter)*images_g + t_inter*images_r
                    interpolate.requires_grad = True
                    Dvals_i = self.run_D(interpolate, gen_c)
                    
                    # Compute gradient penalty
                    gradients = autograd.grad(outputs      = Dvals_i, 
                                              inputs       = interpolate,
                                              grad_outputs = torch.ones_like(Dvals_i),
                                              create_graph = True)[0]
                    gradients = gradients.reshape(gradients.size(0), -1)
                    grad_pen = (torch.clamp(gradients.norm(2, dim=1) - self.gp_clamp, min =0)**2).mean() * self.gp_lambda
                
                # Backpropagate gradient penalty. Note this is much slower is self.gp_source == 'images', because then
                # computing the gradient penalty requires backpropagating through the feature network.
                grad_pen.backward()
                training_stats.report('Loss/D/Grad_Penalty', grad_pen)






################################################
### Original loss function for Projected GAN ###
################################################

class ProjectedGANLoss(Loss):
    def __init__(self, device, G, D, G_ema, blur_init_sigma=0, blur_fade_kimg=0, **kwargs):
        super().__init__()
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        img = self.G.synthesis(ws, c, update_emas=False)
        return img

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())

        logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg', 'Greg']: return  # no regularization needed for PG

        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0

        if do_Gmain:

            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img = self.run_G(gen_z, gen_c)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                loss_Gmain = (-gen_logits).mean()

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()

        if do_Dmain:

            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img = self.run_G(gen_z, gen_c, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                loss_Dgen = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.backward()

            # Dmain: Maximize logits for real images.
            with torch.autograd.profiler.record_function('Dreal_forward'):
                real_img_tmp = real_img.detach().requires_grad_(False)
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                loss_Dreal = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

            with torch.autograd.profiler.record_function('Dreal_backward'):
                loss_Dreal.backward()
