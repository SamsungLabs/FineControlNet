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


from data_generation.constant import setting_labels, instance_descs

preprocessor = None

model_name = 'control_v11p_sd15_openpose'
model = create_model(f'./models/{model_name}.yaml').cpu()
model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def get_2d_control(input_image, detect_resolution, det):
    global preprocessor

    if 'Openpose' in det:
        if not isinstance(preprocessor, OpenposeDetector):
            preprocessor = OpenposeDetector()

    with torch.no_grad():
        input_image = HWC3(input_image)
        
        if det == 'None':
            raise NotImplementedError("Specify the detection type")
        else:
            pose, detected_map = preprocessor(resize_image(input_image, detect_resolution), hand_and_face='Full' in det)
            detected_map = HWC3(detected_map)

            body_joints = pose['bodies']['candidate']
            body_subsets = pose['bodies']['subset']  #  number of persons
            num_persons = len(body_subsets)

            persons = []
            for si, subset in enumerate(body_subsets):
                person = {'bodies': {'candidate': pose['bodies']['candidate'], 'subset': [pose['bodies']['subset'][si]]}}
                if 'Full' in det:
                    if len(pose['hands']) > 2*si+1:
                        person['hands'] = pose['hands'][2*si:2*si+2]
                    else:
                        person['hands'] = []
                        
                    if len(pose['faces']) > si:
                        person['faces'] = [pose['faces'][si]]
                    else:
                        person['faces'] = []
                else:
                    person['hands'] = []
                    person['faces'] = []
                    
                persons.append(person)
    
    return persons, detected_map


def process(det, persons_detections, fusion_type, harmony_level, mask_kernel_size, mask_blur, mask_softmax_temperature, input_image, prompt_list, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, eta):
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
        
        for person_id, person_detection in enumerate(persons_detections):

            """ Draw skeleton """
            if 'Full' not in det:
                person_detection['hands'] = []
                person_detection['faces'] = []

            person_detection = copy.deepcopy(person_detection)
            input_mask = np.zeros_like(img[:, :, :1])
            # draw each instance skeleton
            detected_map = draw_pose(person_detection, H, W)
            detected_map = HWC3(detected_map)
            # draw global skeleton
            local_detection_mask = detected_map.sum(2) > 0
            global_detection_map[local_detection_mask] = detected_map[local_detection_mask]
            """ Draw skeleton """

            """ Sanitize joint pixel coordinates to get a bounding box """
            joints_img = []
            body_joints = person_detection['bodies']['candidate']
            body_subset = person_detection['bodies']['subset'][0]
            for num in body_subset[:18]:
                if num != -1:
                    joints_img.append(body_joints[int(num)][:2])
            
            if 'Full' in det:
                hand_joints = person_detection['hands']
                face_joints = person_detection['faces']
                if len(hand_joints) > 0:
                    for hj in hand_joints:
                        for joint in hj:
                            if joint[0] > 0.01:
                                joints_img.append(joint)
                if len(face_joints) > 0:
                    for joint in face_joints[0]:
                        if joint[0] > 0.01:
                            joints_img.append(joint)

            joints_img = np.array(joints_img, dtype=np.float32)
            joints_valid = np.ones_like(joints_img[:, 0]) 
            joints_img[:, 0] *= W
            joints_img[:, 1] *= H

            bbox = get_bbox_from_joints(joints_img, joints_valid, (H, W), expand_ratio=1.2) 
            x,y,w,h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            """ Sanitize joint pixel coordinates to get a bounding box """


            """ Parse instance masks """
            # bounding box mask
            # input_mask[y:y+h, x:x+w] = 255
            # skeleton mask
            input_mask[detected_map.sum(2) >  0] = 255
            # dilation
            kernel = np.ones((mask_kernel_size, mask_kernel_size), np.uint8)
            input_mask = cv2.dilate(input_mask, kernel)
            
            # mask_pixel: 1. visible
            mask_pixel = cv2.resize(input_mask, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            mask_pixel = cv2.GaussianBlur(mask_pixel, (0, 0), mask_blur)
            mask_latent = cv2.resize(mask_pixel, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
            
            # mask: 1. visible. occupied. 0. not visible. not occupied
            mask = torch.from_numpy(mask_latent.copy()).float().cuda()
            mask = torch.stack([mask for _ in range(num_samples)], dim=0)
            mask = einops.rearrange(mask, 'b h w -> b 1 h w').clone()

            mask_tensor_list.append(mask)
            """ Parse instance masks """


            """ Parse 2D control """
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            control_tensor_list.append(control)
            """ Parse 2D control """

            """ Parse prompts """
            positive_prompt = multi_identity_text_prompt[person_id] + ' ' + person_setting + ', ' + a_prompt
            text_tensor_list.append(model.get_learned_conditioning([positive_prompt]))
            n_text_tensor_list.append(model.get_learned_conditioning([n_prompt]))
            """ Parse prompts """
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
            num_humans = len(persons_detections)
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
    parser.add_argument('--save_dir', type=str, default='./')
    parser.add_argument('--img_path', type=str, default='./test_imgs/standing.jpg')
    parser.add_argument('--fusion_type', type=str, default='h-control', help='pick from h-control, h-all, h-ediff-i, m. empty string for ControlNet.')
    parser.add_argument('--num_persons', type=int, default=2)
    parser.add_argument('--person_scale', type=float, default=1.0)
    parser.add_argument('--crowdedness', type=float, default=0.0, help='0: sufficient distance between skeletons, 1: all skeletons at the middle') 

    parser.add_argument('--seed', type=int, default=1)

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
    a_prompt = 'best quality, visually amusing' 
    n_prompt = 'lowres, bad anatomy, bad hands, bad faces, cropped, worst quality'  # "Negative Prompt"
    num_samples = 1 # 50
    guess_mode = False
    strength = 1.5
    scale = 9.0  # guidance scale # max 30,  min 0.1
    eta = 0.0
    
    # Main hyperparemeters for FineControlNet
    image_resolution = 768
    detect_resolution = 512
    det = "Openpose_Full"  # ["Openpose_Full", "Openpose", "None"]
    ddim_steps = 40
    harmony_level = ddim_steps // 2 # higher more harmony. but trade-off with identity observance. used in attention level
    mask_blur = 1 
    mask_kernel_size = image_resolution // 8
    mask_kernel_size = mask_kernel_size if mask_kernel_size % 2 == 1 else mask_kernel_size + 1 
    mask_softmax_temperature = 0.001 # lower -> more distinct boundary between instances during denoising. higher -> more blurry boundary between instances
    
    ##### Ablation
    num_persons = args.num_persons
    person_scale = args.person_scale
    crowdedness = args.crowdedness
    #####

    print(f"[Options]: fusion_type {fusion_type} hardmony level {harmony_level} det {det} ddim_steps {ddim_steps} mask_kernel_size {mask_kernel_size} mask_temperature {mask_softmax_temperature}")
    print(f"[Ablation]: num_persons {num_persons} person_scale {person_scale} crowdedness {crowdedness}")

    # get reference pose
    input_image = cv2.imread(reference_image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    src_poses, global_detection_map = get_2d_control(input_image, detect_resolution, det)

    # stretch generation image template according to the number of persons
    img_height, img_width = input_image.shape[:2]
    input_image = np.zeros((img_height, img_width * num_persons), dtype=np.uint8)
    new_img_height, new_img_width = input_image.shape[:2]

    # poses is a single element list
    new_poses = [copy.deepcopy(src_poses[0]) for _ in range(num_persons)]
    for person_idx, pose in enumerate(new_poses):
        body_joints = pose['bodies']['candidate']

        num_body_joints = len(body_joints)
        if 'Full' in det:
            hand_joints = pose['hands']
            face_joints = pose['faces']

            if len(hand_joints) > 0:
                for hidx, hj in enumerate(hand_joints): # 21
                    for joint in hj: 
                        body_joints.append(joint)
            
            if len(face_joints) > 0:
                for joint in face_joints[0]: # 70
                    body_joints.append(joint)

        body_joints = np.array(body_joints)
        body_joints[:, 0] *= img_width
        body_joints[:, 1] *= img_height 

        # rescale 
        body_joints[:, 0] /= new_img_width
        body_joints[:, 1] /= new_img_height

        body_joints = body_joints * person_scale

        # 8, 11 hips
        r_hip, l_hip = body_joints[8], body_joints[11]
        pelvis = (r_hip + l_hip) / 2.
        body_joints -= pelvis[None, :]

        # relocate
        delta_distance = (num_persons - (2*person_idx + 1)) * crowdedness
        body_joints += np.array([[1/(2 * num_persons) * (2*person_idx + 1 + delta_distance), 0.5]], dtype=np.float32) 
        body_joints = np.concatenate([body_joints, np.ones_like(body_joints[:, :1])], axis=1)

        if 'Full' in det:
            if len(hand_joints) > 0:
                hand_joints = body_joints[num_body_joints:num_body_joints+2*21, :2].reshape(2,21,2)
                new_poses[person_idx]['hands'] = hand_joints.tolist()

            if len(face_joints) > 0:
                face_joints = body_joints[num_body_joints+2*21:, :2].reshape(1,70,2)
                new_poses[person_idx]['faces'] = face_joints.tolist()
            body_joints = body_joints[:num_body_joints]
            
        new_poses[person_idx]['bodies']['candidate'] = body_joints.tolist()
    poses = new_poses

    multi_identity_text_prompt = random.choices(instance_descs, k=len(poses))
    person_setting = random.choice(setting_labels)
    global_text_prompt = ' and '.join(multi_identity_text_prompt) + ' ' + person_setting

    print("[Global Text Prompt]: ", global_text_prompt)
    print("[Identity Text Prompt]: ", multi_identity_text_prompt)
    print("[Setting Text Prompt]: ", person_setting)
    print("[Pose source file]: ", osp.basename(reference_image_path))
    print("[Seed]: ", seed)

    ips = [det, poses, fusion_type, harmony_level, mask_kernel_size, mask_blur, mask_softmax_temperature, input_image, [multi_identity_text_prompt, person_setting, global_text_prompt], a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, eta]
    results = process(*ips)
    global_detection_map = results[0]
    
    for idx in range(num_samples):
        output = results[1+idx]
        
        detection_path = f'finecontrolnet_detection_{idx}.png'
        detection_path = osp.join(save_dir, detection_path)
        cv2.imwrite(detection_path, global_detection_map[:, :, ::-1])
        output_path = f'finecontrolnet_output_{idx}.png'
        output_path = osp.join(save_dir, output_path)
        cv2.imwrite(output_path, output[:, :, ::-1])
        
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(poses) + 2)]
        output = output.copy()
        cv2.putText(output, person_setting, (15, output.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (5, 5, 5 ))

        for person_id, person_detection in enumerate(poses):
            person_detection = copy.deepcopy(person_detection)
            detected_map = draw_pose(person_detection, output.shape[0], output.shape[1], color=colors[person_id])
            detected_map = HWC3(detected_map)
            tmp = detected_map.sum(2) >  0
            output[tmp] = detected_map[tmp] * 0.9
            cv2.putText(output, multi_identity_text_prompt[person_id], (15, 15*(1+person_id)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (colors[person_id][0] * 255, colors[person_id][1] *255, colors[person_id][2] *255 ))        
            
        output_path = f'finecontrolnet_annotoverlaid_{idx}.png'
        output_path = osp.join(save_dir, output_path)
        cv2.imwrite(output_path, output[:, :, ::-1])
        print("[File Saved To]: ", output_path)

