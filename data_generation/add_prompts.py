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
"""

import sys
import copy
import json
import os.path as osp

import random
import numpy as np

from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything

from pycocotools.coco import COCO
from tqdm import tqdm



MAX_NUM_PEOPLE = 1000 #maximum number of skeletons per image

instance_descs = [
    "A woman with a green dress",       # Woman/Man
    "A woman in a yellow shirt",
    "A woman wearing a purple skirt",
    "A woman in a pink dress",
    "A man in a grey shirt",
    "A man with a green jacket",
    "A man wearing a yellow sweater",
    "A man in a purple shirt",
    "A man with a pink jacket",
    "A woman in an orange shirt",
    "A woman with a turquoise dress",
    "A woman in a beige skirt",
    "A woman wearing a white dress",
    "A man in a navy blue shirt",
    "A man with an olive green sweater",
    "A man wearing a maroon shirt",
    "A man in a teal jacket",
    "A woman in a silver dress",
    "A woman with a brown skirt",
    "A woman in a light blue shirt",
    "A woman wearing a black dress",
    "A man in a white shirt",
    "A man with a burgundy jacket",
    "A man in a charcoal grey sweater",
    "A man wearing a tan jacket",
    'A chef',                          # Jobs/Roles
    'A police officer',
    'A firefighter',
    'A pilot',
    'A doctor',
    'A nurse',
    'A construction worker',
    'A judge',
    'A farmer',
    'A scientist',
    'A artist',
    'A detective',
    'A librarian',
    'A ballerina',
    'A photographer',
    'A gardener',
    'A sailor',
    'A mechanic',
    'A postman',
    'A knight in armor',
    'An astronaut',
    'A racecar driver',
    'A king with a crown',
    'A queen with a crown',
]

# Simple descriptions

setting_descs = [
    "in a parking lot",
    "on the surface of the moon",
    "at a carnival",
    "at a birthday party",
    "at sunset",
    "at a wedding",
    "on a tropical beach",
    "in a dense forest",
    "on top of a snowy mountain",
    "inside a bustling city subway",
    "in a quiet library",
    "at an airport",
    "on a deserted island",
    "in the desert",
    "in a medieval castle",
    "on a space station",
    "in a museum",
    "on the deck of a cruise ship",
    "in an ancient temple",
    "at a state fair",
    "inside a high-tech laboratory",
    "on a safari in the African savannah",
    "at a vibrant street market",
    "at a baseball game",
    "in a park",
    "on a rooftop",
    "in Times Square",
]


seed_total = random.randint(0,2147483647)
seed_everything(seed_total)


data_split = 'val'
db = COCO(f'coco_{data_split}_pose_data_finecontrolnet.json')

new_img_id, new_annot_id = 0, 0
new_images, new_annotations = [], []


for iid in tqdm(db.imgs.keys()):

    img = db.imgs[iid]
    aid = db.getAnnIds([iid])[0] # one annotation per image.
    ann = db.anns[aid]

    imgname = osp.join(f'{data_split}2017', img['file_name'])
    img['file_name']
    file_name = img['file_name']
    img_width, img_height = img['width'], img['height'] 
    poses = ann['people']['poses'] 

    num_people = ann["img_num_people"]
    

    if num_people <= MAX_NUM_PEOPLE:
        seed = random.randint(0,2147483647)

        setting = random.choice(setting_descs)

        instances = []
        for pose in poses:
            instances.append(random.choice(instance_descs))

        new_img_dict = copy.deepcopy(img)
        new_img_dict['id'] = new_img_id 
        new_images.append(new_img_dict)


        # determine centerpoints of each person
        centerpoints_x = []
        for pose in poses:
            locations = pose["bodies"]["candidate"]
            center_x = np.mean(np.asarray(locations), 0)[:-2]
            centerpoints_x.append(center_x[0])

        # reorder left to right
        # reorder instance_desc left to right
        pairs = sorted(zip(centerpoints_x, instances))

        # Unzip the pairs
        sorted_centerpoints_x, sorted_instances = zip(*pairs)

        # Convert the tuples back to lists, if needed
        sorted_centerpoints_x = list(sorted_centerpoints_x)
        sorted_instances = list(sorted_instances)

        instances_reordered = sorted_instances

        # make global prompt for 2 people
        if len(poses) == 2:
            global_prompt = instances_reordered[0] + " on the left and " + instances_reordered[1] + " on the right " + setting

        # make global prompt for 3 people
        elif len(poses) == 3:
            global_prompt = instances_reordered[0] + " on the left, " + instances_reordered[1] + " in the middle, and " + instances_reordered[2] + " on the right " + setting

        # make global prompt for 3 people
        elif len(poses) > 3:
            global_prompt = "From left to right: "
            for idx, instance_desc in enumerate(instances_reordered):
                if idx == len(instances_reordered) - 2:
                    global_prompt += instance_desc + ", and "
                elif idx == len(instances_reordered) - 1:
                    global_prompt += instance_desc + " "
                else:
                    global_prompt += instance_desc + ", "
            global_prompt += setting
            

        else:
            print("ERROR: Not enough poses", len(poses))

        global_prompt = global_prompt.replace(" A ", " a ")

        
        new_ann_dict = copy.deepcopy(ann)
        new_ann_dict['id'] = new_annot_id 
        new_ann_dict['image_id'] = new_img_id

        #
        new_ann_dict['global_desc'] = global_prompt 
        new_ann_dict['instance_descs'] = instances 
        new_ann_dict['setting_desc'] = setting
        new_ann_dict['seed'] = seed 
        #
        new_annotations.append(new_ann_dict)

        new_img_id += 1
        new_annot_id += 1


output = {'images': new_images, 'annotations': new_annotations}
with open(f'coco_{data_split}_pose_with_prompt_data_finecontrolnet.json', 'w') as f:
    json.dump(output, f)