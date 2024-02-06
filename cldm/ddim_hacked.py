"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.
Author(s):
Hongsuk Choi (redstonepo@gmail.com)
Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE file.

Modified from ControlNet: https://github.com/lllyasviel/ControlNet
"""


"""SAMPLING ONLY."""
from collections import defaultdict, Counter

import torch
import torch.nn.functional as F
import torchvision

import numpy as np
import random
import copy
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               
               mask_tensor_list=[],
               mask_softmax_temperature=1.0,
               global_cond=None,
               global_un_cond=None,
               harmony_level=6,
               fusion_type='',
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule,
                                                    
                                                    mask_tensor_list=mask_tensor_list,
                                                    mask_softmax_temperature=mask_softmax_temperature,
                                                    global_cond=global_cond,
                                                    global_un_cond=global_un_cond,
                                                    harmony_level=harmony_level,
                                                    fusion_type=fusion_type,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None,
                      
                      mask_tensor_list=[],
                      mask_softmax_temperature=1.0,
                      global_cond=None,
                      global_un_cond=None,
                      harmony_level=6,
                      fusion_type=''
                      ):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        # [951, 901, 851, 801, 751, 701, 651, 601, 551, 501, 451, 401, 351, 301, 251, 201, 151, 101,  51,  1]

     
        ddpm_steps_per_ddim_step = self.ddpm_num_timesteps // len(self.ddim_timesteps)
        # print(f"Time range: {time_range}, ddpm steps per ddim step: {ddpm_steps_per_ddim_step}")

        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1 # originally total steps is 20
            
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

                
            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]


            if fusion_type != '': 
                # occupancy mask to attention masks
                if fusion_type == 'h-ediff-i':
                    modified_mask_tensor_list = [mask_tensor + 1 / (1e12 * mask_tensor.sum()) for mask_tensor in mask_tensor_list]
                    mask_argmax = torch.stack(modified_mask_tensor_list).argmax(dim=0) 
                    hard_mask_tensor_list = torch.stack([mask_argmax == hi for hi in range(len(mask_tensor_list))]).float()
                    new_mask_tensor_list = hard_mask_tensor_list
                            
                elif i < len(iterator) - harmony_level: # harmony level 0. always hard masking. harmony level == len(iterator). no hard masking
                    # [1, 1, 64, 96]
                    # in case of overlap between instances, more weight on smaller instance mask for balance. negligble?
                    modified_mask_tensor_list = [mask_tensor + 1 / (1e12 * mask_tensor.sum()) for mask_tensor in mask_tensor_list]
                    mask_argmax = torch.stack(modified_mask_tensor_list).argmax(dim=0) 
                    hard_mask_tensor_list = torch.stack([mask_argmax == hi for hi in range(len(mask_tensor_list))]).float()
                    
                    soft_mask_tensor_list = F.softmax(torch.stack(mask_tensor_list) / mask_softmax_temperature, dim=0)
                    # fill foreground with hard mask
                    soft_mask_tensor_list[:, sum(mask_tensor_list) != 0] = hard_mask_tensor_list[:, sum(mask_tensor_list) != 0]
                    new_mask_tensor_list = soft_mask_tensor_list
                    # vis
                    # if i == 0:
                    #     for jj, mask_tensor in enumerate(new_mask_tensor_list):
                    #         torchvision.utils.save_image(mask_tensor, f'check{jj}.png')

                else:
                    # [num_humans, 1, 1, 64, 96]
                    new_mask_tensor_list = F.softmax(torch.stack(mask_tensor_list) / mask_softmax_temperature, dim=0)
                
                outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,

                                        dynamic_threshold=dynamic_threshold,
                                        
                                        mask_tensor_list=new_mask_tensor_list,
                                        global_cond=global_cond,
                                        global_un_cond=global_un_cond,
                                        # use_global_cond=use_global_cond,
                                        fusion_type=fusion_type
                                        )
                img, pred_x0 = outs

                if fusion_type == 'x':
                    # img, pred_x0: (b*num_humans, 4, h, w)
                    img, pred_x0 = img.reshape(-1, len(new_mask_tensor_list), *img.shape[1:]), pred_x0.reshape(-1, len(new_mask_tensor_list), *pred_x0.shape[1:])
                    #(num_humans, b, 4, h, w)
                    img, pred_x0 = img.transpose(0,1), pred_x0.transpose(0,1)
                    # new_mask_tensor_list: (num_humans, b, 1, h, w)
                    img, pred_x0 = (img * new_mask_tensor_list).sum(0), (pred_x0 * new_mask_tensor_list).sum(0)

            else:
                outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                        quantize_denoised=quantize_denoised, temperature=temperature,
                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                        corrector_kwargs=corrector_kwargs,
                        unconditional_guidance_scale=unconditional_guidance_scale,
                        unconditional_conditioning=unconditional_conditioning,
                        dynamic_threshold=dynamic_threshold,
                        )

                img, pred_x0 = outs
     
                
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None,
                      
                      mask_tensor_list=[],
                      global_cond=None,
                      global_un_cond=None,
                      use_global_cond=False,
                      fusion_type=''
                      ):
        b, *_, device = *x.shape, x.device
        
        if not use_global_cond and len(mask_tensor_list) > 0 and fusion_type != '':
            ##########################
            # Hack
            # c_concat: control, c_crossattn: text
            # c['c_concat']: always and originally one length list. c['c_concat'][0] is control_tensor_list; one control_tensor shape is b,c,h,w. b is the number of samples
            # c['c_crossatn']: length is the number of samples. c['c_crossatn'][sample_idx] is text_tensor_list; one text_tensor shape is 1,77,768. 768==w. 1 and 77 seem to be fixed
            # for unconditoinal_conditioning, it's same
            
            # Original Format
            # c_concat: length 1 list;  b,c,h,w. b is the number of samples
            # c_crossattn: length 1 list; b,c',w. b is the number of samples 
            ##########################

            num_humans = len(mask_tensor_list)
            
            if fusion_type != '' and fusion_type != 'm': # 'x', 'h-all', 'h-control', 'h-ediff-i' 
                if fusion_type == 'h-ediff-i':
                    noise_level = torch.Tensor([index]).to(x)
                    mask_weight = 0.4 * torch.log(1 + noise_level**2).item()
                else:
                    mask_weight = 1
                
                new_x = x[:, None].expand(b, num_humans, -1, -1, -1)
                new_x = new_x.reshape(b*num_humans, *new_x.shape[2:])
                new_t = t[:, None].expand(b, num_humans)
                new_t = new_t.reshape(b*num_humans, *new_t.shape[2:])

                if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                    model_output = self.model.apply_model(new_x, new_t, c, mask=mask_tensor_list, fusion_type=fusion_type, mask_weight=mask_weight, )
                else:
                    model_t = self.model.apply_model(new_x, new_t, c, mask=mask_tensor_list, fusion_type=fusion_type, mask_weight=mask_weight,)
                    model_uncond = self.model.apply_model(new_x, new_t, unconditional_conditioning, mask=mask_tensor_list, mask_weight=mask_weight, fusion_type=fusion_type)
                    model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)
                
                e_t = model_output

            # MultiControlNet - https://github.com/huggingface/diffusers/tree/multi_controlnet
            # https://github.com/huggingface/diffusers/blob/multi_controlnet/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_controlnet.py
            elif fusion_type == 'm':
                
                new_x = x[:, None].expand(b, num_humans, -1, -1, -1)
                new_x = new_x.reshape(b*num_humans, *new_x.shape[2:])
                new_t = t[:, None].expand(b, num_humans)
                new_t = new_t.reshape(b*num_humans, *new_t.shape[2:])
                
                if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                    model_output = self.model.apply_model(new_x, new_t, c, mask=mask_tensor_list, fusion_type=fusion_type, 
                    global_text_embedding=global_cond["c_crossattn"]
                    )
                else:
                    model_t = self.model.apply_model(new_x, new_t, c, mask=mask_tensor_list, fusion_type=fusion_type,
                    global_text_embedding=global_cond["c_crossattn"]
                    )
                    model_uncond = self.model.apply_model(new_x, new_t, unconditional_conditioning, mask=mask_tensor_list, fusion_type=fusion_type,
                    global_text_embedding=global_un_cond["c_crossattn"])
                    model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

                e_t = model_output
        
        else:
            if use_global_cond and global_cond is not None and global_un_cond is not None:
                c, unconditional_conditioning = global_cond, global_un_cond

            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                model_output = self.model.apply_model(x, t, c)
            else:
                model_t = self.model.apply_model(x, t, c)
                model_uncond = self.model.apply_model(x, t, unconditional_conditioning)
                model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        
        if fusion_type == 'x':
            b *= num_humans
            x = new_x
            t = new_t

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        num_reference_steps = timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), timesteps[i], device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec
