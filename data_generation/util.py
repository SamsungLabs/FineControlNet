import numpy as np

from typing import Dict

def data_condition_checker(people: Dict[str, list], num_persons=2):
    # condition
    poses = people['poses']
    crowd_scores = people['crowd_scores']
    res_ratios = people['res_ratios']

    skip = False
    # two people
    if len(poses) != num_persons:
        skip = True

    for idx in range(len(poses)):
        if (np.array(poses[idx]['bodies']['subset']) != -1).sum() < 10:
            skip = True
            break

    # similar resolutions
    res_ratio_ratio_thr = 0.8
    min_res_ratio, max_res_ratio = min(res_ratios), max(res_ratios)
    if min_res_ratio / max_res_ratio < res_ratio_ratio_thr:
        skip = True

    # skip already too small human images, where they are likely to be far away
    if max_res_ratio < 0.1:
        skip = True

    # low crowdIndex
    min_crowd_score = min(crowd_scores)
    crowd_score_thr = 0.2
    if min_crowd_score > crowd_score_thr:
        skip = True
    
    return skip


def compute_CrowdIndex(ref_bbox, ref_kps, intf_kps):

    na = 0
    for ref_kp in ref_kps:
        count = get_inclusion(ref_bbox, ref_kp)
        na += count

    nb = 0
    for intf_kp in intf_kps:
        count = get_inclusion(ref_bbox, intf_kp)
        nb += count

    if na < 4:  # invalid ones, e.g. truncated images
        return 0
    else:
        return nb / na


def get_inclusion(bbox, kp):
    if bbox[0] > kp[0] or (bbox[0] + bbox[2]) < kp[0]:
        return 0

    if bbox[1] > kp[1] or (bbox[1] + bbox[3]) < kp[1]:
        return 0

    return 1


def compute_iou(src_roi, dst_roi):
    # IoU calculate with GTs
    xmin = np.maximum(dst_roi[:, 0], src_roi[:, 0])
    ymin = np.maximum(dst_roi[:, 1], src_roi[:, 1])
    xmax = np.minimum(dst_roi[:, 0] + dst_roi[:, 2], src_roi[:, 0] + src_roi[:, 2])
    ymax = np.minimum(dst_roi[:, 1] + dst_roi[:, 3], src_roi[:, 1] + src_roi[:, 3])

    interArea = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    boxAArea = dst_roi[:, 2] * dst_roi[:, 3]
    boxBArea = np.tile(src_roi[:, 2] * src_roi[:, 3], (len(dst_roi), 1))
    sumArea = boxAArea + boxBArea

    iou = interArea / (sumArea - interArea + 1e-5)

    return iou


def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint



def add_neck(coco_joints_name, joint_coord):
    lhip_idx = coco_joints_name.index('L_Shoulder')
    rhip_idx = coco_joints_name.index('R_Shoulder')
    neck = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
    neck[2] = joint_coord[lhip_idx,2] * joint_coord[rhip_idx,2] # joint_valid
    neck = neck.reshape(1, 3)
    joint_coord = np.concatenate((neck, joint_coord))
    return joint_coord

def parse_to_controlnet_img_shape(H, W, resolution):
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    return H, W


def get_bbox_from_joints(joint_img, joint_valid, img_shape, expand_ratio=1.2):
    h, w = img_shape
    # joint_img: (J, 2)
    # joint_valid: (J,)

    x_img, y_img = joint_img[:,0], joint_img[:,1]
    x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5*width*expand_ratio
    xmax = x_center + 0.5*width*expand_ratio
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5*height*expand_ratio
    ymax = y_center + 0.5*height*expand_ratio

    # sanitize
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w-1, xmax)
    ymax = min(h-1, ymax)

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox


def sanitize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 > x1 and y2 > y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        bbox = None

    return bbox

def process_bbox(bbox, img_width, img_height, target_width, target_height):
    bbox = sanitize_bbox(bbox, img_width, img_height)
    if bbox is None:
        return bbox

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = target_width / target_height
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    
    bbox = bbox.astype(np.float32)
    return bbox