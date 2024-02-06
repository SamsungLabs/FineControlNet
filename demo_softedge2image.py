from share import *
import config
import os
import os.path as osp
from pathlib import Path
import glob
import argparse
import copy
import json
from tqdm import tqdm

from pycocotools.coco import COCO
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector, draw_pose
from annotator.openpose.util import get_bbox_from_joints
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

from annotator.hed import HEDdetector
from annotator.pidinet import PidiNetDetector


def get_bbox_from_edges(input_edge, img_shape, expand_ratio=1.2):
    input_edge[:10,:] = [0,0,0]
    input_edge[-10:,:] = [0,0,0]
    input_edge[:,-10:] = [0,0,0]
    input_edge[:,:10] = [0,0,0]

    B = np.argwhere(input_edge > 10)
    (ystart, xstart, zstart), (ystop, xstop, zstop) = B.min(0), B.max(0) + 1 
    bbox = np.array([xstart, ystart, xstop, ystop]).astype(np.float32)

    return bbox


preprocessor = None

model_name = 'control_v11p_sd15_mlsd'  # somehow 'control_v11p_sd15_softedge' does not work as well as mlsd... - Hongsuk & Isaac
model = create_model(f'./models/{model_name}.yaml').cpu()
model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def process(det, input_edges, fusion_type, harmony_level, mask_kernel_size, mask_blur, mask_softmax_temperature, input_image, prompt_list, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, eta):
    input_image = HWC3(input_image)
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape
    print("[Generation Image shape]: ", H, W)

    multi_identity_text_prompt, person_setting, global_text_prompt = prompt_list
    if global_text_prompt == '':
        global_text_prompt = ' and '.join(multi_identity_text_prompt) + ' ' + person_setting
        global_text_prompt = global_text_prompt.lower()
    global_detection_map = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    with torch.no_grad():
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        ###################################################
        # Obtain a (text,2D control) pair for each person #
        ###################################################
        control_tensor_list = []
        mask_tensor_list = []
        text_tensor_list = []
        n_text_tensor_list = []
        total_forground_mask_pixel = np.zeros_like(img[:, :, 0], dtype=np.float32)
        for instance_id, input_edge in enumerate(input_edges):


            input_edge = copy.deepcopy(input_edge)
            input_mask = np.zeros_like(img[:, :, :1])


            bbox = get_bbox_from_edges(input_edge, (H, W), expand_ratio=1.2)  # x,y,w,h

            x,y,w,h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
  
            input_mask[y:h, x:w] = 255

            # dilation
            kernel = np.ones((mask_kernel_size, mask_kernel_size), np.uint8)
            input_mask = cv2.dilate(input_mask, kernel)
            
            # mask_pixel: 1. visible
            mask_pixel = cv2.resize(input_mask, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            mask_pixel = cv2.GaussianBlur(mask_pixel, (0, 0), mask_blur)
            mask_latent = cv2.resize(mask_pixel, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
            
            # try background condtion
            total_forground_mask_pixel += mask_pixel

            # mask: 1. visible. occupied. 0. not visible. not occupied
            mask = torch.from_numpy(mask_latent.copy()).float().cuda()
            mask = torch.stack([mask for _ in range(num_samples)], dim=0)
            mask = einops.rearrange(mask, 'b h w -> b 1 h w').clone()

            input_edge = cv2.resize(input_edge, (W, H), interpolation=cv2.INTER_LINEAR)

            control = torch.from_numpy(input_edge.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            
            # list
            control_tensor_list.append(control)
            # for semantically harmonized instance with background (person_setting). mountain -> mountain sports apparel or thick coat
            positive_prompt = multi_identity_text_prompt[instance_id] + ' ' + person_setting + ', ' + a_prompt
            # positive_prompt = multi_identity_text_prompt[person_id] + ', ' + a_prompt

            text_tensor_list.append(model.get_learned_conditioning([positive_prompt]))
            # print("prompt: ", positive_prompt)

            # Don't do this. ex) woman and woman. woman - woman = neutral or man. soldier and cyborg. cyborg - soldier = ? usually people imagine solider like cyborg images.
            # negative_prompt = other identities + ', ' + n_prompt
            negative_prompt = n_prompt

            n_text_tensor_list.append(model.get_learned_conditioning([negative_prompt]))
            # print("n prompt: ",negative_prompt)
            mask_tensor_list.append(mask)
        ###################################################
        # Obtain a (text,2D control) pair for each person #
        ###################################################
        
        # list of (text,2D control) pairs
        cond = {"c_concat": [control_tensor_list], "c_crossattn": [text_tensor_list] * num_samples}
        un_cond = {"c_concat": None if guess_mode else [control_tensor_list], "c_crossattn": [n_text_tensor_list] * num_samples}

        # reshape conditions
        if fusion_type != '':
            c_concat_list = []
            c_crossattn_list = []
            uc_concat_list = []
            uc_crossattn_list = []
            num_humans = len(input_edges)
            for idx in list(range(num_humans)):
                c_concat = cond["c_concat"][0][idx] # b,c,h,w.
                c_concat_list.append(c_concat)
                
                c_crossattn = torch.stack([x[idx][0] for x in cond["c_crossattn"]], dim=0)  # b, c', w
                c_crossattn_list.append(c_crossattn)
                
                uc_concat = un_cond["c_concat"][0][idx] # b,c,h,w.
                uc_concat_list.append(uc_concat)
                
                uc_crossattn = torch.stack([x[idx][0] for x in un_cond["c_crossattn"]], dim=0)  # b, c', w
                uc_crossattn_list.append(uc_crossattn)
            
            c_concat = torch.stack(c_concat_list).transpose(0,1) # b,n,c,h,w
            c_crossattn = torch.stack(c_crossattn_list).transpose(0,1) # b,n,c',w
            c_concat = c_concat.reshape(num_samples*num_humans, *c_concat.shape[2:]) # b*n,c,h,w
            c_crossattn = c_crossattn.reshape(num_samples*num_humans, *c_crossattn.shape[2:]) # b*n,c',w
            
            uc_concat = torch.stack(uc_concat_list).transpose(0,1) # b,n,c,h,w
            uc_crossattn = torch.stack(uc_crossattn_list).transpose(0,1) # b,n,c',w
            uc_concat = uc_concat.reshape(num_samples*num_humans, *uc_concat.shape[2:]) # b*n,c,h,w
            uc_crossattn = uc_crossattn.reshape(num_samples*num_humans, *uc_crossattn.shape[2:]) # b*n,c',w
            
            cond = {"c_concat": [c_concat], "c_crossattn": [c_crossattn]}
            un_cond = {"c_concat": [uc_concat], "c_crossattn": [uc_crossattn]}

        # Global condition for ControlNet and MultiControlnet
        global_detection_map = cv2.resize(global_detection_map, (W, H), interpolation=cv2.INTER_LINEAR)
        global_control = torch.from_numpy(global_detection_map.copy()).float().cuda() / 255.0
        global_control = torch.stack([global_control for _ in range(num_samples)], dim=0)
        global_control = einops.rearrange(global_control, 'b h w c -> b c h w').clone()
        global_cond = {"c_concat": [global_control], "c_crossattn": [model.get_learned_conditioning([global_text_prompt + ', ' + a_prompt] * num_samples)]}
        global_un_cond = {"c_concat": None if guess_mode else [global_control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        if fusion_type == '':
            cond = global_cond
            un_cond = global_un_cond

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)
        
        shape = (4, H // 8, W // 8)
        # From ControlNet: Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                shape, cond, verbose=False, eta=eta,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=un_cond,

                                                mask_tensor_list=mask_tensor_list,
                                                mask_softmax_temperature=mask_softmax_temperature,
                                                global_cond=global_cond,
                                                global_un_cond=global_un_cond,
                                                harmony_level=harmony_level,
                                                fusion_type=fusion_type,
                                                )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(num_samples)]
        
    return [global_detection_map] + results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./test_outputs/softedge')
    parser.add_argument('--img_path', type=str, default='./test_imgs/example_softedge/', help='Direcotry to source images or edges.')
    parser.add_argument('--fusion_type', type=str, default='h-control', help='pick from h-control, h-all, h-ediff-i, m. empty string for ControlNet.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--instance_descriptions', nargs='*', default=["a tiger cub", "a gray wolf"], help='List of N instance descriptions.')
    parser.add_argument('--setting_descriptions', type=str, default='in the jungle', help='Description of the setting.')
    parser.add_argument('--xy', nargs='*', type=lambda s: tuple(map(int, s.split(','))), default=[(50,280), (275,300)], help='List of x,y locations as tuples (x,y).')
    parser.add_argument('--wh', nargs='*', type=lambda s: tuple(map(int, s.split(','))), default=[(200,150), (225,140)], help='List of width,height dimensions as tuples (w,h).')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # argument parse and create log
    args = parse_args()
    reference_image_path = args.img_path
    save_dir = args.save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fusion_type = args.fusion_type  # '', 'h-ediff-i', 'h-all', 'h-control', 'm'
    seed = args.seed
    seed_everything(seed)

    # ControlNet hyperparameters
    a_prompt = 'best quality' 
    n_prompt = 'lowres, blur, bad anatomy, bad face, worst quality'  # "Negative Prompt"
    num_samples = 1 # 50
    guess_mode = False
    strength = 1.5
    scale = 15.0  # guidance scale # max 30,  min 0.1
    eta = 0.0
    
    # Main hyperparemeters for FineControlNet
    image_resolution = 512
    detect_resolution = 1024
    det = "SoftEdge_PIDI"
    ddim_steps = 30
    harmony_level = ddim_steps #// 2 # higher more harmony. but trade-off with identity observance. used in attention level
    mask_blur = 1 
    mask_kernel_size = 1 
    mask_kernel_size = mask_kernel_size if mask_kernel_size % 2 == 1 else mask_kernel_size + 1 
    mask_softmax_temperature = 0.001 # lower -> more distinct boundary between instances during denoising. higher -> more blurry boundary between instances

    print(f"[Options]: fusion_type {fusion_type} hardmony level {harmony_level} det {det} ddim_steps {ddim_steps} mask_kernel_size {mask_kernel_size} mask_temperature {mask_softmax_temperature}")

    input_img_path_list = sorted(glob.glob(osp.join(reference_image_path, '*.jpg'))) 
    input_img_path_list.extend(sorted(glob.glob(osp.join(reference_image_path, '*.png'))) )
    input_img_path_list.extend(sorted(glob.glob(osp.join(reference_image_path, '*.jpeg'))) )

    input_edges = []
    for input_idx_imgs, input_img_path in enumerate(tqdm(input_img_path_list)):
    
        control_image = cv2.imread(input_img_path) #
        control_image = cv2.cvtColor(control_image, cv2.COLOR_BGR2RGB)
        control_image = cv2.resize(control_image, (args.wh[input_idx_imgs]))

        x,y = args.xy[input_idx_imgs]

        input_image = np.ones((image_resolution, image_resolution, 3)) * 255
        input_image = input_image.astype(np.uint8)


        if y + control_image.shape[0] <= input_image.shape[0] and x + control_image.shape[1] <= input_image.shape[1]:
            # Place the image
            input_image[y:y+control_image.shape[0], x:x+control_image.shape[1]] = control_image
        else:
            print("Foreground image does not fit within background dimensions.")

        if 'HED' in det:
            if not isinstance(preprocessor, HEDdetector):
                preprocessor = HEDdetector()

        if 'PIDI' in det:
            if not isinstance(preprocessor, PidiNetDetector):
                preprocessor = PidiNetDetector()

        with torch.no_grad():
            input_image = HWC3(input_image)

            if det == 'None':
                detected_map = input_image.copy()
            else:
                detected_map = preprocessor(resize_image(input_image, detect_resolution), safe='safe' in det)
                detected_map = HWC3(detected_map)

            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape

            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        input_edges.append(detected_map)

    multi_identity_text_prompt = args.instance_descriptions
    person_setting = args.setting_descriptions
    global_text_prompt = ' and '.join(multi_identity_text_prompt) + ' ' + person_setting

    # Save total canny
    print("[Global Text Prompt]: ", global_text_prompt)
    print("[Identity Text Prompt]: ", multi_identity_text_prompt)
    print("[Setting Text Prompt]: ", person_setting)
    print("[Pose source file]: ", osp.basename(reference_image_path))
    print("[Seed]: ", seed)

    ips = [det, input_edges, fusion_type, harmony_level, mask_kernel_size, mask_blur, mask_softmax_temperature, input_image, [multi_identity_text_prompt, person_setting, global_text_prompt], a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, eta]
    results = process(*ips)
    global_detection_map, output = results[0], results[1]
    
    detection_path = f'finecontrolnet_detection.png'
    detection_path = osp.join(save_dir, detection_path)

    combined_image = np.zeros_like(input_edges[0])

    # Overlay each image
    for img in input_edges:
        mask = np.any(img > 0, axis=-1)
        combined_image[mask] = img[mask]
    
    cv2.imwrite(detection_path, combined_image)
    output_path = f'finecontrolnet_output.png'
    output_path = osp.join(save_dir, output_path)
    cv2.imwrite(output_path, output[:, :, ::-1])
    print("[File Saved To]: ", output_path)

