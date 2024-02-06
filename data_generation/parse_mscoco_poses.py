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


import copy
import json
import os
import os.path as osp
import argparse
import sys

import cv2
import numpy as np

from pycocotools.coco import COCO
from tqdm import tqdm
from util import *




class MSCOCO:
    def __init__(self, data_dir, data_split, resolution) -> None:
        data_dir = data_dir  # /labdata/hongsuk/MSCOCO/2017
        self.annot_path = osp.join(data_dir, 'annotations')
        self.img_path = osp.join(data_dir, 'images')
        self.data_split = data_split # 'train' or 'val'

        self.img_resolution = resolution
        self.min_num_body_joints = 6

        self.joint_set = {
                            'joint_num': 134, # body 24 (23 + pelvis), lhand 21, rhand 21, face 68
                            'joints_name': \
                                ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', # body part
                                'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand
                                'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', # right hand
                                *['Face_' + str(i) for i in range(56,73)], # face contour
                                *['Face_' + str(i) for i in range(5,56)] # face
                                ),
                            'flip_pairs': \
                                ((1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16) , (18,21), (19,22), (20,23), # body part
                                (24,45), (25,46), (26,47), (27,48), (28,49), (29,50), (30,51), (31,52), (32,53), (33,54), (34,55), (35,56), (36,57), (37,58), (38,59), (39,60), (40,61), (41,62), (42,63), (43,64), (44,65), # hand part
                                (66,82), (67,81), (68,80), (69,79), (70,78), (71,77), (72,76), (73,75), # face contour
                                (83,92), (84,91), (85,90), (86,89), (87,88), # face eyebrow
                                (97,101), (98,100), # face below nose
                                (102,111), (103,110), (104,109), (105,108), (106,113), (107,112), # face eyes
                                (114,120), (115,119), (116,118), (121,125), (122,124), # face mouth
                                (126,130), (127,129), (131,133) # face lip
                                )
                        }
        # https://github.com/jin-s13/COCO-WholeBody#what-is-coco-wholebody
        self.coco_body_joints_name = ('Neck', 'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel')

        # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/80d4c5f7b25ba4c3bf5745ab7d0e6ccd3db8b242/.github/media/keypoints_pose_18.png
        self.openpose_body_joints_name = ('Nose', 'Neck', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip',  'L_Knee', 'L_Ankle', 'R_Eye', 'L_Eye', 'R_Ear',  'L_Ear')


        self.datalist = self.load_data()

    def merge_joint(self, joint_img, feet_img, lhand_img, rhand_img, face_img):
        # pelvis
        lhip_idx = self.joint_set['joints_name'].index('L_Hip')
        rhip_idx = self.joint_set['joints_name'].index('R_Hip')
        pelvis = (joint_img[lhip_idx,:] + joint_img[rhip_idx,:]) * 0.5
        pelvis[2] = joint_img[lhip_idx,2] * joint_img[rhip_idx,2] # joint_valid
        pelvis = pelvis.reshape(1,3)
        
        # feet
        lfoot = feet_img[:3,:]
        rfoot = feet_img[3:,:]

        joint_img = np.concatenate((joint_img, pelvis, lfoot, rfoot, lhand_img, rhand_img, face_img)).astype(np.float32) # self.joint_set['joint_num'], 3
        return joint_img


    def load_data(self):
        db = COCO(osp.join(self.annot_path, f'coco_wholebody_{self.data_split}_v1.0.json'))
        # db = COCO(osp.join(self.annot_path, f'person_keypoints_{self.data_split}2017.json')) # body

        images = []; annotations = []
        img_id = 0; annot_id = 0;
        
        for iid in tqdm(db.imgs.keys()):
            img = db.imgs[iid]
            aids = db.getAnnIds([iid])

            people = {'poses': [], 'res_ratios': [], 'crowd_scores' : [], 'ious': []} # 
            
            # if 1 person skip
            if len(aids) < 2:
                continue

            for aid_idx, aid in enumerate(aids):
                ann = db.anns[aid]
                # img = db.loadImgs(ann['image_id'])[0]
                imgname = osp.join(f'{self.data_split}2017', img['file_name'])
        
                if ann['iscrowd'] or (ann['num_keypoints'] == 0):
                    continue

                # Expected height and width of the ControlNet input
                target_height, target_width = parse_to_controlnet_img_shape(img['height'], img['width'], self.img_resolution)

                # bbox
                bbox = process_bbox(ann['bbox'], img['width'], img['height'], target_width, target_height) 
                if bbox is None: continue
                    
                # joint coordinates
                joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1,3)
                foot_img = np.array(ann['foot_kpts'], dtype=np.float32).reshape(-1,3)
                lhand_img = np.array(ann['lefthand_kpts'], dtype=np.float32).reshape(-1,3)
                rhand_img = np.array(ann['righthand_kpts'], dtype=np.float32).reshape(-1,3)
                face_img = np.array(ann['face_kpts'], dtype=np.float32).reshape(-1,3)
                joint_img = self.merge_joint(joint_img, foot_img, lhand_img, rhand_img, face_img)

                joint_valid = (joint_img[:,2].copy().reshape(-1,1) > 0).astype(np.float32)

                joint_img[:,2:] = joint_valid

                # use body annotation to fill hand/face annotation
                for body_name, part_name in (('L_Wrist', 'L_Wrist_Hand'), ('R_Wrist', 'R_Wrist_Hand'), ('Nose', 'Face_18')):
                    if joint_valid[self.joint_set['joints_name'].index(part_name),0] == 0:
                        joint_img[self.joint_set['joints_name'].index(part_name)] = joint_img[self.joint_set['joints_name'].index(body_name)]
                        joint_valid[self.joint_set['joints_name'].index(part_name)] = joint_valid[self.joint_set['joints_name'].index(body_name)]

                # new bbox
                tight_bbox = get_bbox_from_joints(joint_img, joint_valid[:, 0], (img['height'], img['width']), expand_ratio=1.2)
                bbox = process_bbox(tight_bbox, img['width'], img['height'], target_width, target_height) 
                if bbox is None: continue

                
                ### conver the joints to the openpose format ###
                # normalize to 0~1
                joint_img[:, 0] /= img['width']
                joint_img[:, 1] /= img['height']
                
                # parse body joints 
                coco_body_joint_img = joint_img[:24, :]  # 24 joints
                # insert the neck location at the first
                coco_body_joint_img = add_neck(self.joint_set['joints_name'], coco_body_joint_img) 
                # transform
                openpose_body_joint_img = transform_joint_to_other_db(coco_body_joint_img, self.coco_body_joints_name, self.openpose_body_joints_name)

                # filter out if not enough body joints
                if (openpose_body_joint_img[:, 2] == 1).sum() < self.min_num_body_joints:
                    continue

                # parse hand joints
                coco_right_hand_joint_img = joint_img[24:45, :] # check
                coco_left_hand_joint_img = joint_img[45:66, :] # check

                # no need transform for hand and face keypoints. they are the same
                # COCO Whole Body: https://github.com/jin-s13/COCO-WholeBody#what-is-coco-wholebody
                # OpenPose Full (Whole) Body: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/80d4c5f7b25ba4c3bf5745ab7d0e6ccd3db8b242/.github/media/keypoints_hand.png , pose18, pose28, face
                openpose_right_hand_joint_img = coco_right_hand_joint_img
                openpose_left_hand_joint_img = coco_left_hand_joint_img 
            
                # parse face joints
                coco_face_joint_img = joint_img[66:]
                # little bit different. but order dosen't matter. actually number is also little bit different
                openpose_face_joint_img = coco_face_joint_img

                # for single person
                subset = np.arange(20)
                subset_18 = subset[:18]
                subset_18[openpose_body_joint_img[:, 2] == 0] = -1

                subset[:18] = subset_18
                openpose_pose_format = {
                        'bodies': {'candidate': openpose_body_joint_img.tolist(),
                                'subset': [subset.tolist()]
                                }
                }

                # all or not
                # if ann['righthand_valid'] and ann['lefthand_valid']:
                if np.all(openpose_right_hand_joint_img[:, 2] > 0) and  np.all(openpose_left_hand_joint_img[:, 2] > 0):
                    openpose_pose_format['hands'] = [openpose_right_hand_joint_img[:, :2].tolist(), openpose_left_hand_joint_img[:, :2].tolist()]
                else:
                    openpose_pose_format['hands'] = []
                
                
                # if ann['face_valid'] :
                if np.all(openpose_face_joint_img[:, 2] > 0):
                    openpose_pose_format['faces'] = [openpose_face_joint_img[:, :2].tolist()]
                else:
                    openpose_pose_format['faces'] = []

                # visualize
                # canvas = draw_pose(openpose_pose_format,  img['height'],  img['width'])
                # cv2.imwrite(f'check_{osp.basename(img_path)}', canvas[:, :, ::-1])
                # print(f'check_{osp.basename(img_path)}')
                people['poses'].append(openpose_pose_format)

                """ resolution """
                # area = ann['area']
                bbox_over_full_img = (tight_bbox[2] * tight_bbox[3]) / (img['width'] * img['height'])
                people['res_ratios'].append(bbox_over_full_img)

                """ crowd idx """ 
                other_aids = aids[:aid_idx] + aids[aid_idx+1:]
                ref_bbox = np.array(ann['bbox'], dtype=np.float32)
                ref_joint = np.array(ann['keypoints'], dtype=np.float32).reshape(-1,3)
                crowd_idx_list = []

                for oaid in other_aids:
                    other_ann = db.anns[oaid]
                    other_bbox = np.array(other_ann['bbox'], dtype=np.float32)
                    other_joint = np.array(other_ann['keypoints'], dtype=np.float32).reshape(-1,3)

                    iou = compute_iou(ref_bbox[None, :], other_bbox[None, :])[0, 0] 
                    crowd_idx = compute_CrowdIndex(ref_bbox, ref_joint, other_joint)

                    crowd_idx_list.append(crowd_idx)
                
                total_crowd_idx = np.sum(crowd_idx_list) # total crowd_idx with other instance; could be bigger than 1 

                people['crowd_scores'].append(total_crowd_idx.item())

            # skip one person image. 
            if len(people['poses']) < 2:
                continue

            annot_dict = {}
            annot_dict['id'] = annot_id
            annot_dict['image_id'] = iid  # don't change
            annot_dict['people'] = people

            annot_dict['img_crowdidx'] = np.mean(people['crowd_scores']) 
            annot_dict['img_num_people'] = len(people['poses'])

            annotations.append(annot_dict)
            annot_id += 1
            
            # just use the original img dict
            images.append(img)
            # img_id += 1

        print("total num images: ", len(images), len(annotations))
        output = {'images': images, 'annotations': annotations}
        return output
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str)
    parser.add_argument('--subset', type=str, default='val')
    parser.add_argument('--resolution', type=float, default=512)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    data_dir, data_split, resolution = args.datadir, args.subset, args.resolution  #'/labdata/hongsuk/MSCOCO/2017', 'val', 512
    dataset = MSCOCO(data_dir, data_split, resolution)

    output_path = f'coco_{data_split}_pose_data_finecontrolnet.json'
    with open(output_path, 'w') as f:
        json.dump(dataset.datalist, f)
    print("Saved to ", output_path)

    vis = False
    # Visualize
    if vis:
        sys.path.append(osp.abspath(osp.join(__file__, '..', '..'))) #'/home/hongsuk.c/Projects/ControlNet-v1-1-nightly')
        print(osp.abspath(osp.join(__file__, '..', '..')))
        from annotator.openpose import draw_pose
        data_split = 'val'
        db = COCO(f'coco_{data_split}_pose_data_finecontrolnet.json')
        for iid in tqdm(db.imgs.keys()):
            img = db.imgs[iid]
            aid = db.getAnnIds([iid])[0] # one annotation per image.
            ann = db.anns[aid]

            # variable you might use
            imgname = osp.join(f'{data_split}2017', img['file_name'])
            img['file_name']
            file_name = img['file_name']
            img_width, img_height = img['width'], img['height'] 
            poses = ann['people']['poses'] 

            for pid, pose in enumerate(poses):
                # visualize pose; openpose_pose_format
                canvas = draw_pose(pose,  img['height'],  img['width'])
                cv2.imwrite(f'check_{osp.basename(imgname)[:-4]}_{pid}.png', canvas[:, :, ::-1])
                print(f'check_{imgname}')
