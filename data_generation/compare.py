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


import os
import os.path as osp
import glob
import shutil



output_dir = '/labdata/selim/finecontrol_baselines'
save_dir = '/home/hongsuk.c/Downloads/for_patent'

selected_idx_list =  [0,1,19,52,92,103,198,253] 

for idx in selected_idx_list:
    outputs = glob.glob(osp.join(output_dir, 'output*.png'))

    for dir in glob.glob(output_dir + '/*'):
        if 'finecontrolnet' in dir:
            continue
        
        if 'unicontrol' in dir:
            method_name  = osp.basename(dir)
            img_name = f'{idx}.png'
            shutil.copy(osp.join(dir, img_name), osp.join(save_dir, f'{method_name}_{idx:08d}.jpg'))

        else:
            method_name = osp.basename(dir)
            img_name = f'{method_name}_{idx:08d}.jpg'
            shutil.copy(osp.join(dir, img_name), osp.join(save_dir, img_name))


    

