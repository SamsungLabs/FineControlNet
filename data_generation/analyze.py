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

from pycocotools.coco import COCO
import os
import os.path as osp
import glob
import shutil
from tqdm import tqdm
from collections import defaultdict
import numpy as np



dataset_path = './parsed_coco_val_finecontrolnet_with_prompts.json'


output_dir = '/home/hongsuk.c/Downloads/finecontrolnet_qualitative' #'./output/benchmark_h4'
outputs = glob.glob(osp.join(output_dir, '*.png'))
# outputs = glob.glob(osp.join(output_dir, 'detecion*.png'))

outputs = {osp.basename(x): x for x in outputs}

# copy_dir = './output/benchmark_h4_copy'

db = COCO(dataset_path)
count  = 0
miss_list = []
iid_list = []
num_people_stats = defaultdict(int)
for iid in tqdm(db.imgs.keys()):
    img = db.imgs[iid]
    aid = db.getAnnIds([iid])[0] # one annotation per image.
    ann = db.anns[aid]

    imgname = osp.join(f'val2017', img['file_name'])
    file_name = img['file_name']
    

    people = ann['people']
    poses = people['poses']
    crowd_scores = people['crowd_scores']
    res_ratios = people['res_ratios']
    # condition

    # two people
    if len(poses) != 4:
        continue
    
    for idx in range(len(poses)):
        if (np.array(poses[idx]['bodies']['subset']) != -1).sum() < 10:
            skip = True
            break

    # similar resolutions
    res_ratio_ratio_thr = 0.1 #0.8
    min_res_ratio, max_res_ratio = min(res_ratios), max(res_ratios)
    if min_res_ratio / max_res_ratio < res_ratio_ratio_thr:
        continue

    if max_res_ratio < 0.1: #0.1:
        continue

    # low crowdIndex
    min_crowd_score = min(crowd_scores)
    crowd_score_thr = 0.2
    if min_crowd_score > crowd_score_thr:
        continue
    
    iid_list.append(iid)
    
    # result_name = f'finecontrolnet_output{iid}.png'
    # if result_name in outputs:
    #     print(iid, ann['global_desc'])
    #     iid_list.append(iid)


    # // find missing people.
    # if file_name[:-4] not in outputs:
    #     count += 1
    #     miss_list.append(file_name)

    # // get num people stats
    # poses = ann['people']['poses']
    # num_people_stats[len(poses)] += 1
    
    # //copy to a new iid name
    # if file_name[:-4] not in outputs:
    #     continue
    # new_name = f'finecontrolnet_output{iid}.png'#f'detecion_{file_name[:-4]}_{prompt_descriptor}_seed{seed}.png'
    # shutil.copy(outputs[file_name[:-4]], osp.join(copy_dir, new_name))

    # # // print the prompt
    # prompt = ann['global_desc']
    # if iid in [0,1,19,52,92,103,198,253]:
    #     print(iid, prompt)

print(len(iid_list), iid_list)
    
# print(sorted(num_people_stats.items()))