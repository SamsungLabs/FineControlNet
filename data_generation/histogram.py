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
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

dataset_path = './parsed_coco_val_finecontrolnet_new.json'

# copy_dir = './output/benchmark_h4_copy'

db = COCO(dataset_path)
count  = 0




num_people_list = []
person_resolution_list = []
person_crowdidx_list = []
for iid in tqdm(db.imgs.keys()):
    img = db.imgs[iid]
    aid = db.getAnnIds([iid])[0] # one annotation per image.
    ann = db.anns[aid]

    imgname = osp.join(f'val2017', img['file_name'])
    file_name = img['file_name']
    
    people = ann['people']

    person_res = people['res_ratios']
    person_crowdidx = people['crowd_scores']
    num_people = ann['img_num_people']

    num_people_list.append(num_people)
    person_resolution_list.extend(person_res)
    person_crowdidx_list.extend(person_crowdidx)

# crop the data
person_crowdidx_list = np.array(person_crowdidx_list)
person_crowdidx_list[person_crowdidx_list > 1.0] = 1.0


n_bins = np.arange(2,16,2) 
fig, ((ax0, ax1)) = plt.subplots(nrows=1, ncols=2)

ax0.hist(num_people_list, bins=n_bins, density=False, color=['tan'])
# ax0.legend(prop={'size': 8})
# ax0.set_title('Number of persons per image')
ax0.set_xlabel('Number of persons per image')
ax0.set_xticks(n_bins)

import pdb; pdb.set_trace()
n_bins = np.arange(0,12,2) /10.
colors = [(255/255,127/255,127/255), (127/255,127/255,255/255)]#['tan', 'lime']
labels = ['Resolution ratio', 'CrowdIndex']
x = np.asarray([person_resolution_list, person_crowdidx_list]).T  
ax1.hist(x, bins=n_bins, density=False, histtype='bar', color=colors, label=labels)
ax1.legend(prop={'size': 8})
# ax0.set_title('Resolution ratio & CrowdIndex per person')
ax1.set_xlabel('Ratio value per person')
ax1.set_xticks(n_bins)


fig.tight_layout()
plt.show()

print("CHECK")