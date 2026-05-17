# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
import time

# ---------------------------------------------------------
# Helper: safe squared norm mean (keeps dtype stable)
def _mean_sq_norm(x):
    # x: [B, ...]
    return (x.float().view(x.shape[0], -1).pow(2).sum(dim=1)).mean()

# ---------------------------------------------------------
    

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, cur_tick, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, sparse_weight=0.1, mask_sparse_weight=0.0):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.sparse_weight = sparse_weight
        self.mask_sparse_weight = mask_sparse_weight
        self.recon_loss = torch.nn.MSELoss()

        # Jacobian estimation parameters
        self.num_noise_probes = 4  # You can adjust this hyperparameter
        self.eps = 1e-8

        print(f'Using StyleGAN2 loss with r1_gamma={r1_gamma}, pl_weight={pl_weight}, pl_decay={pl_decay}, pl_batch_shrink={pl_batch_shrink}, style_mixing_prob={style_mixing_prob}, augment_pipe={augment_pipe}.')

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            if self.sparse_weight > 0:
                content, style = self.G_mapping.encode_latents(z, c)
                ws = self.G_mapping.get_ws_using_latents(content, style)
            else:
                ws = self.G_mapping(z, c)
                content, style = None, None
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]

        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws, content, style

    # Subash Added for jacobian estimation using hutchinson's method
    def run_G_for_jacobian_finite_difference(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            content, style = self.G_mapping.encode_latents(z, c)
            ws = self.G_mapping.get_ws_using_latents(content, style)

            content_delta = torch.randn_like(content) * self.epsilon_cotent
            style_delta = torch.randn_like(style) * self.epsilon_style

            ws_content_perturbed = self.G_mapping.get_ws_using_latents(content + content_delta, style)
            ws_style_perturbed = self.G_mapping.get_ws_using_latents(content, style + style_delta)

            #Stack ws along batch dimension
            ws_all = torch.cat([ws, ws_content_perturbed, ws_style_perturbed], dim=0)

        with misc.ddp_sync(self.G_synthesis, sync):
            img_all = self.G_synthesis(ws_all)
            # Split images back to original and perturbed
            img, img_content_perturbed, img_style_perturbed = torch.split(img_all, z.shape[0], dim=0) 

        return img, img_content_perturbed, img_style_perturbed, style
    
    def run_G_fd_content(self, z, c, fd_eps, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            content, style = self.G_mapping.encode_latents(z, c)
            ws = self.G_mapping.get_ws_using_latents(content, style)

            content_delta = torch.randn_like(content) * fd_eps

            ws_content_perturbed = self.G_mapping.get_ws_using_latents(content + content_delta, style)

            #Stack ws along batch dimension
            ws_all = torch.cat([ws, ws_content_perturbed], dim=0)

        with misc.ddp_sync(self.G_synthesis, sync):
            img_all = self.G_synthesis(ws_all)
            # Split images back to original and perturbed
            img, img_content_perturbed = torch.split(img_all, z.shape[0], dim=0) 

        return img, img_content_perturbed, content, style

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits
    

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, cur_tick, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws, gen_content, gen_style = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.

                #Reconstruction of noise and style and content encoders
                loss_c_recon = self.recon_loss(self.G_mapping.content_inverse_encoder(gen_content), gen_z[:, self.G_mapping.i_dim:])
                loss_s_recon = self.recon_loss(self.G_mapping.style_inverse_encoder(gen_style, gen_c), gen_z[:, :self.G_mapping.i_dim])

                #report
                training_stats.report('Loss/G/content_recon', loss_c_recon)
                training_stats.report('Loss/G/style_recon', loss_s_recon)

                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Ggan = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/gan', loss_Ggan)
                
                loss_Gmain = loss_Ggan + 0.001 * (loss_c_recon + loss_s_recon)
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply orthogonal regularization. path length regularization is replaced with orthogonal regularizaiton
        if do_Gpl:

            with torch.autograd.profiler.record_function('jacobian_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                
                #Create labels gen_c for smaller batch with uniform sampling of labels
                gen_c_uniform = torch.zeros((batch_size, gen_c.shape[1]), device=self.device)
                num_classes = gen_c.shape[1]
                
                if batch_size % num_classes != 0:
                    samples_per_class = batch_size // num_classes
                    for i in range(num_classes - 1):
                        gen_c_uniform[i * samples_per_class: (i + 1) * samples_per_class, i] = 1.0
                    #last class
                    gen_c_uniform[(num_classes - 1) * samples_per_class:, num_classes - 1] = 1.0
                else:
                    samples_per_class = batch_size // num_classes
                    for i in range(num_classes):
                        gen_c_uniform[i * samples_per_class: (i + 1) * samples_per_class, i] = 1.0


                gen_img, _, gen_content, gen_style = self.run_G(gen_z[:batch_size], gen_c_uniform, sync=sync)
                
                
                content_style_matrix = torch.tensor(0.0, device=self.device)
                vjp_jacobian_content = torch.tensor(0.0, device=self.device)
                vjp_jacobian_style = torch.tensor(0.0, device=self.device)
               

                for k in range(self.num_noise_probes):
                    vjp_noise_k = torch.randn_like(gen_img)/np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                    with torch.autograd.profiler.record_function(f'content_style_grads_{k}'), conv2d_gradfix.no_weight_gradients():
                        vjp_jac_content, vjp_jac_style = torch.autograd.grad(
                            outputs=gen_img,
                            inputs=[gen_content, gen_style],
                            grad_outputs=vjp_noise_k,
                            create_graph=True,
                            retain_graph=True,
                        )
                    
                    content_style_matrix = content_style_matrix + torch.einsum('bi,bj->bij', vjp_jac_style, vjp_jac_content)

                    vjp_jacobian_content = vjp_jacobian_content + vjp_jac_content.pow(2).sum(dim=1)
                    vjp_jacobian_style = vjp_jacobian_style + vjp_jac_style.pow(2).sum(dim=1)

                content_style_matrix = content_style_matrix / self.num_noise_probes

                content_style_orthogonality_loss = content_style_matrix.square().sum(dim=(1, 2))

                content_jacobian_loss = vjp_jacobian_content / self.num_noise_probes
                style_jacobian_loss = vjp_jacobian_style / self.num_noise_probes

                content_style_orthogonality_normalized = (content_style_orthogonality_loss / (content_jacobian_loss * style_jacobian_loss + self.eps)).mean()


                training_stats.report('Loss/G/content_style_orthogonality', content_style_orthogonality_loss.mean())
                training_stats.report('Loss/G/content_jacobian_norm', content_jacobian_loss.mean())
                training_stats.report('Loss/G/style_jacobian_norm', style_jacobian_loss.mean())
                training_stats.report('Loss/G/content_style_orthogonality_normalized', content_style_orthogonality_normalized)

            with torch.autograd.profiler.record_function('jacobian_backward'):
                (gen_img[:, 0, 0, 0] * 0 + 2.0 * content_style_orthogonality_normalized ).mean().mul(gain).backward()

                

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws, gen_content, gen_style = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
