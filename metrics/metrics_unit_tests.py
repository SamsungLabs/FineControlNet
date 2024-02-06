import torch
import numpy as np
from PIL import Image
import torchvision.transforms as TF
from omegaconf import OmegaConf
import argparse


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image',
                        type=str,
                        default='/home/selim/Downloads/two_person/orig/twoperson3.jpg',
                        help='path of the input image for the unit tests')

    parser.add_argument('--gt_pose',
                        type=str,
                        default='/home/selim/Downloads/two_person/pose/results_twoperson3.npz',
                        help='path of the GT pose for the pose accuracy unit test')

    parser.add_argument('--all',
                        action='store_true',
                        help='run all tests if true')

    parser.add_argument('--text',
                        action='store_true',
                        help='run the text (clip) score tests if true')

    parser.add_argument('--quality',
                        action='store_true',
                        help='run the quality (fid and kid) score tests if true')

    parser.add_argument('--pose',
                        action='store_true',
                        help='run the pose score tests if true')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.all or args.text:
        from .text_metrics import TextMetrics
        # CLIP similarity unit tests #
        x = Image.open(args.image)
        xx = torch.tensor(np.array(x)).permute(2, 0, 1)[None]
        bboxes = [[[80, 80, 270, 407], [308, 30, 472, 407]]]

        text_metric = TextMetrics(device='cpu')
        y1 = text_metric.compute_local_clip_score([["lady in yellow shirt", "guy in green shirt"]], bboxes, xx)
        y2 = text_metric.compute_local_clip_score([["lady in red shirt", "panda in blue pants"]], bboxes, xx)
        print('Local CLIP scores', y1, y2)

        y3 = text_metric.compute_global_clip_score(["two people in an office"], xx)
        y4 = text_metric.compute_global_clip_score(["two cats in a home"], xx)
        print('Global CLIP scores', y3, y4)

    if args.all or args.quality:
        from .quality_metrics import QualityMetrics
        # Image quality unit tests #

        normalized = True
        compute_for_fakes = True

        quality_metric = QualityMetrics(device='cpu', normalized=normalized)
        x = Image.open(args.image)
        xx = quality_metric.image_transforms(x)[None]
        print('image shape:', xx.shape)

        if normalized:
            fake_images = torch.randn(25, 3, 299, 299)
        else:
            fake_images = torch.randint(0, 255, (25, 3, 299, 299), dtype=torch.uint8)

        if compute_for_fakes:
            y5 = quality_metric.compute(fake_images)
        else:
            y5 = quality_metric.compute(xx.repeat(25, 1, 1, 1))

        real_or_fake = 'fake' if compute_for_fakes else 'real'
        print(f'Image quality metric for {real_or_fake}:', y5)

    if args.all or args.pose:
        from .pose_metrics import PoseMetrics
        # Pose accuracy unit tests #

        config = OmegaConf.load("metrics/metrics.yaml")
        mmpose_config_file = config['params']['pose']['mmpose_config_file']
        mmpose_checkpoint_file = config['params']['pose']['mmpose_checkpoint_file']

        pose_metric = PoseMetrics(device='cpu',
                                  mmpose_config_file=mmpose_config_file,
                                  mmpose_checkpoint_file=mmpose_checkpoint_file,
                                  )

        x1 = Image.open(args.image)
        x1_np = np.array(x1)

        gt_pose = pose_metric.predict_pose(x1_np)

        data = {
            "pose": [gt_pose]
        }

        y6 = pose_metric.compute(data, x1_np[None])
        print("Pose accuracy metric:", y6)
