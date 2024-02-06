"""Main API for computing and reporting evaluation metrics. Adapted from https://github.com/IDEA-Research/HumanSD"""

import os
import time
import json
from pycocotools.coco import COCO
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import argparse
import glob
from omegaconf import OmegaConf

from .utils import instantiate_from_config, get_bboxes_from_poses, aggregate_results, get_mmpose_dict, overlay_skeleton_on_image


class EvaluationMetrics:
    def __init__(self,
                 device,
                 metrics,
                 pose,
                 quality,
                 text
                 ) -> None:

        self.device = device
        # self.metrics = metrics
        self.text = text
        self.quality = quality
        self.pose = pose

    def cli_metrics(self, args):
        metrics = []
        if args.text:
            metrics.append("text")
        if args.quality:
            metrics.append("quality")
        if args.pose:
            metrics.append("pose")

        self.metrics = metrics
        print('self.metrics')
        self.init_metrics(metrics)
        return metrics

    def init_metrics(self, metrics):
        if "text" in metrics:
            from .text_metrics import TextMetrics
            self.text_metrics = TextMetrics(self.device, **self.text)

        if "quality" in metrics:
            from .quality_metrics import QualityMetrics
            self.quality_metrics = QualityMetrics(self.device, **self.quality)

        if "pose" in metrics:
            from .pose_metrics import PoseMetrics
            self.pose_metrics = PoseMetrics(self.device, **self.pose)

    def evaluation(self, data, pred_images, mode):
        results_dict = {}

        if "text" in self.metrics and mode == 'text':
            global_text_prompts = data["global_text_prompts"]
            global_text_score = self.text_metrics.compute_global_clip_score(global_text_prompts, pred_images)

            local_text_prompts = data["local_text_prompts"]
            local_bounding_boxes = data["local_bounding_boxes"]
            local_text_score = self.text_metrics.compute_local_clip_score(
                local_text_prompts, local_bounding_boxes, pred_images)

            pred_image_pil = Image.fromarray(pred_images[0])
            global_text_score_openai = self.text_metrics.compute_global_clip_score_openai(
                global_text_prompts, pred_image_pil)

            local_text_score_openai = self.text_metrics.compute_local_clip_score_openai(
                local_text_prompts, local_bounding_boxes, pred_image_pil)

            results_dict = {**global_text_score, **local_text_score,
                            **global_text_score_openai, **local_text_score_openai}

        if "quality" in self.metrics and mode == 'quality':
            quality_scores = self.quality_metrics.compute(pred_images)
            results_dict = {**results_dict, **quality_scores}

        if "pose" in self.metrics and mode == 'pose':
            pose_scores = self.pose_metrics.compute(data, pred_images)
            results_dict = {**results_dict, **pose_scores}

        return results_dict


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--method_name', type=str)
    parser.add_argument('--text', action='store_true', help='run text metrics if true')
    parser.add_argument('--quality', action='store_true', help='run quality metrics if true')
    parser.add_argument('--pose', action='store_true', help='run pose metrics if true')
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--dataset_path', type=str, default='./parsed_coco_val_finecontrolnet_with_prompts.json')

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    config = OmegaConf.load("metrics/metrics.yaml")
    evaluator = instantiate_from_config(config)
    evaluator.cli_metrics(args)

    results_dir = args.results_dir
    method_name = args.method_name

    db = COCO(args.dataset_path)
    agg_results = dict()
    pred_images = []
    for iid in tqdm(db.imgs.keys()):
        img = db.imgs[iid]
        aid = db.getAnnIds([iid])[0]  # one annotation per image.
        ann = db.anns[aid]
        

        img_width, img_height = img['width'], img['height']
        poses = ann['people']['poses']

        if method_name in ['controlnet', 't2i', 'humansd']:
            pred_image = Image.open(os.path.join(results_dir, f'{method_name}_{iid:08d}.jpg'))
        elif method_name in ['unicontrol', 'diffblender', 'gligen']:
            pred_image = Image.open(os.path.join(results_dir, f'{iid}.png'))
        else:
            try:
                img_coco_id = img['file_name'].split('.')[0]
                fcn_file_name = glob.glob(os.path.join(results_dir, f"output_{img_coco_id}_*"))[0]

            except:
                fcn_file_name = os.path.join(results_dir, f'finecontrolnet_output_{iid}.png')
                # fcn_file_name = os.path.join(results_dir, f'finecontrolnet_fusion_h-v3_output_{iid}.png')

            pred_image = Image.open(fcn_file_name)

        bboxes = get_bboxes_from_poses(poses, pred_image.size[0], pred_image.size[1])
        pred_image_np = np.array(pred_image)[None]
        # overlayed_image = overlay_skeleton_on_image(np.array(pred_image), poses, img_height=pred_image.size[1], img_width=pred_image.size[0])

        # text consistency
        if args.text:
            data = dict(
                global_text_prompts=[ann['global_desc']],
                local_text_prompts=[ann['instance_descs']],
                local_bounding_boxes=[bboxes]
            )

            try:
                results_dict_text = evaluator.evaluation(data, pred_image_np, mode='text')
            except:
                # data['global_text_prompts'] = ["a group of people" + ' ' + ann['setting_desc']]
                data['global_text_prompts'] = [' '.join(ann['global_desc'].split(' ')[:30]) + ' ' + ann['setting_desc']]
                results_dict_text = evaluator.evaluation(data, pred_image_np, mode='text')

            agg_results = aggregate_results(agg_results, results_dict_text)

        # pose consistency
        if args.pose:
            poses_mmlab = [get_mmpose_dict(np.array(instance_pose['bodies']['candidate']),
                                           pred_image.size[1], pred_image.size[0]) for instance_pose in poses]

            data = dict(pose=[poses_mmlab])
            results_dict_pose = evaluator.evaluation(data, pred_image_np, mode='pose')
            agg_results = aggregate_results(agg_results, results_dict_pose)

        # store images for computing image quality scores
        if args.quality:
            pred_image_torch = evaluator.quality_metrics.image_transforms(pred_image)[None]
            pred_images.append(pred_image_torch)

    if args.quality:
        pred_images = torch.concat(pred_images)
        results_dict_quality = evaluator.evaluation(None, pred_images, mode='quality')
        agg_results = aggregate_results(agg_results, results_dict_quality)

    save_name = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))

    print('Method:', method_name)
    for key in sorted(agg_results.keys()):
        val = agg_results[key]
        if 'HND' in key or 'differences' in key:
            result_string = f'{key} metric: {np.mean(val):.3f}, {np.std(val):.3f}'
        else:
            result_string = f'{key} metric: {np.mean(val):.3f}'
        print(result_string)

    metrics_names = '_'.join(evaluator.metrics)
    save_dir = args.save_dir if args.save_dir is not None else 'results'
    os.makedirs(save_dir, exist_ok=True)

    try:
        with open(os.path.join(save_dir, f'{method_name}_{metrics_names}_{save_name}.json'), 'w') as fp:
            json.dump(agg_results, fp)
    except:
        print('Failed to dump to json file!')
