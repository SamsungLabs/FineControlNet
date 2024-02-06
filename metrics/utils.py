import importlib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os.path as osp
import copy

from annotator.openpose import draw_pose
from annotator.util import HWC3, resize_image
from annotator.openpose.util import draw_bodypose


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_bboxes_from_poses(poses, img_width, img_height):
    """Computes bounding boxes per pose in [left, top, right, bottom] convention
    """
    bboxes = []
    for pose in poses:
        keypoints = np.array(pose['bodies']['candidate'])
        valid_keypoints = keypoints[keypoints[:, -1] > 0]
        x_left, x_top = valid_keypoints.min(0)[:2] * np.array([img_width, img_height])
        x_right, x_bottom = valid_keypoints.max(0)[:2] * np.array([img_width, img_height])

        bboxes.append([int(x_left), int(x_top), int(x_right), int(x_bottom)])

    return bboxes


def aggregate_results(agg_results, results_dict):

    for key, val in results_dict.items():
        if val < 0:  # SKIP the negative value, which indicates there was no gt
            continue
        if key not in agg_results:
            agg_results[key] = [val]
        else:
            agg_results[key].append(val)

    return agg_results


def openpose_to_mmlab(present_pose):

    return np.array([
        [present_pose[0, 0], present_pose[0, 1], present_pose[0, 2]],
        [present_pose[15, 0], present_pose[15, 1], present_pose[15, 2]],
        [present_pose[14, 0], present_pose[14, 1], present_pose[14, 2]],
        [present_pose[17, 0], present_pose[17, 1], present_pose[17, 2]],
        [present_pose[16, 0], present_pose[16, 1], present_pose[16, 2]],
        [present_pose[5, 0], present_pose[5, 1], present_pose[5, 2]],
        [present_pose[2, 0], present_pose[2, 1], present_pose[2, 2]],
        [present_pose[6, 0], present_pose[6, 1], present_pose[6, 2]],
        [present_pose[3, 0], present_pose[3, 1], present_pose[3, 2]],
        [present_pose[7, 0], present_pose[7, 1], present_pose[7, 2]],
        [present_pose[4, 0], present_pose[4, 1], present_pose[4, 2]],
        [present_pose[11, 0], present_pose[11, 1], present_pose[11, 2]],
        [present_pose[8, 0], present_pose[8, 1], present_pose[8, 2]],
        [present_pose[12, 0], present_pose[12, 1], present_pose[12, 2]],
        [present_pose[9, 0], present_pose[9, 1], present_pose[9, 2]],
        [present_pose[13, 0], present_pose[13, 1], present_pose[13, 2]],
        [present_pose[10, 0], present_pose[10, 1], present_pose[10, 2]],
    ]).astype(float)


def get_mmpose_dict(poses_op, image_h, image_w):

    keypoints = openpose_to_mmlab(poses_op)
    score = keypoints[:, -1].mean()
    keypoints[:, 0] *= image_w
    keypoints[:, 1] *= image_h

    return {'keypoints': keypoints, 'score': score}


def overlay_poses(idx, poses, multi_identity_text_prompt):
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(poses) + 2)]
    for person_id, person_detection in enumerate(poses):
        person_detection = copy.deepcopy(person_detection)
        # assuming image_resolution == detection_resolution
        detected_map = draw_pose(person_detection, output.shape[0], output.shape[1], color=colors[person_id])
        detected_map = HWC3(detected_map)
        tmp = detected_map.sum(2) > 0
        output[tmp] = detected_map[tmp] * 0.6

        output = output.copy()
        cv2.putText(output, multi_identity_text_prompt[person_id], (15, 15*(1+person_id)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (colors[person_id][0] * 255, colors[person_id][1] * 255, colors[person_id][2] * 255))
    output_path = f'{idx}.png'
    output_path = osp.join('test_results', output_path)
    cv2.imwrite(output_path, output[:, :, ::-1])
    print("[File Saved To]: ", output_path)


def overlay_skeleton_on_image(image, poses, img_height, img_width):
    for pose in poses:
        canvas = draw_pose(pose, img_height, img_width)
        image = overlay_images(image, canvas)

    image = resize_image(image, 512)
    return image


def draw_pose(pose, H, W, canvas=None):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    if canvas is None:
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = draw_bodypose(canvas, candidate, subset)

    return canvas


def overlay_images(img1, img2):
    # Find indices where img2 is non-black
    non_black_indices = np.where(np.any(img2 != 0, axis=-1))

    # Overwrite img1 at these indices with img2
    img1[non_black_indices] = img2[non_black_indices]

    return img1
