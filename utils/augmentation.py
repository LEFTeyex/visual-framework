r"""
Data augmentation utils.
Consist of all the data augmentation methods for using datasets.
"""

import cv2
import math
import random
import numpy as np

__all__ = ['random_affine_or_perspective', 'cutout', 'mixup']


def random_affine_or_perspective(img, label, filter_bbox=True,
                                 perspective=0.0, angle=0.0, scale=0.0, shear=0.0, translate=0.0, flip=0.0):
    r"""
    Random affine or perspective.
    Consist of perspective, rotation, scale, shear, flip.
    Args:
        img: = image shape is (h, w, c)
        label: label shape is (n, 5) if object detection which the format of bbox is xyxy
        filter_bbox: = True/False whether filter bbox after augment
        perspective: (-perspective, perspective) in x and y, level is about (1e3)
        angle: = rotation angle [0, 180]
        scale: scale in (1 - scale, 1 + scale) [0, 1]
        shear: = shear angle [0, 90]
        translate: = scaling of w and h (pixels) [0 , 1]
        flip: the probability for x and y flip [0 , 1]

    Returns:
        img, label
    """
    h, w, _ = img.shape

    # perspective
    p = np.eye(3)
    p[2, 0] = random.uniform(-perspective, perspective)  # x perspective
    p[2, 1] = random.uniform(-perspective, perspective)  # y perspective

    # rotation and scale
    r = np.eye(3)
    angle = random.uniform(-angle, angle)
    scale = random.uniform(1 - scale, 1 + scale)
    r[:2] = cv2.getRotationMatrix2D(angle=angle, center=(w / 2, h / 2), scale=scale)  # todo center to check

    # shear
    s = np.eye(3)
    s[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear
    s[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear

    # flip
    f = np.eye(3)
    if random.uniform(0, 1) < flip:
        f[0, 0] = -1
        f[0, 2] = w - 1
    if random.uniform(0, 1) < flip:
        f[1, 1] = -1
        f[1, 2] = h - 1

    # translation
    t = np.eye(3)
    t[0, 2] = random.uniform(-translate, translate) * w
    t[1, 2] = random.uniform(-translate, translate) * h

    # combine matrix
    m = p @ r @ s @ f @ t
    if perspective:
        img = cv2.warpPerspective(img, m, (w, h))
    else:  # affine
        img = cv2.warpAffine(img, m[:2], (w, h))

    # transform label
    # TODO now only for object detection label but segmentation or instance segmentation
    n = len(label)
    if n:
        xy = np.ones((n * 4, 3))
        # notice all the four xyxy need to transform
        xy[:, :2] = label[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ m.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or

        # calculate new bbox
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        bbox = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        bbox[:, [0, 2]] = bbox[:, [0, 2]].clip(0, w)
        bbox[:, [1, 3]] = bbox[:, [1, 3]].clip(0, h)
        if filter_bbox:
            filter_bbox = filter_bbox_transform(label[:, 1:5].T * scale, bbox.T)
            label = label[filter_bbox]
            label[:, 1:5] = bbox[filter_bbox]
        else:
            label[:, 1:5] = bbox
    return img, label


def filter_bbox_transform(bbox1, bbox2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
    r"""
    Filter bboxes of transform in affine or perspective.
    Args:
        bbox1: = bbox before augment shape (4, n)
        bbox2: = bbox after augment shape (4, n)
        wh_thr: = threshold for wh after augment (pixels)
        ar_thr: = aspect ratio threshold
        area_thr: = area ratio threshold
        eps:  = 1e-16

    Returns:
        filter (ndarray bool)
    """
    w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
    w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # filter


def cutout(img, label):
    r"""
    CutOut augmentation https://arxiv.org/abs/1708.04552.
    Args:
        img: = image shape is (h, w, c)
        label: = label shape is (n, x)

    Returns:
        img, label
    """
    h, w = img.shape[:2]
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8
    for s in scales:
        # create random masks
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        x_min = max(0, random.randint(0, w) - mask_w // 2)
        y_min = max(0, random.randint(0, h) - mask_h // 2)
        x_max = min(w, x_min + mask_w)
        y_max = min(h, y_min + mask_h)

        # zero mask
        img[y_min:y_max, x_min:x_max] = np.array(0, dtype=np.uint8)

        # random color mask
        # img[y_min:y_max, x_min:x_max] = [np.random.randint(64, 191, dtype=np.uint8) for _ in range(3)]

        # TODO maybe need to filter the label which is bad
        # if len(label):
        #     pass

    return img, label


def mixup(img, label, img2, label2, beta=8.0):
    r"""
    MixUp augmentation https://arxiv.org/abs/1710.09412.
    Args:
        img: = image1
        label: = label1 shape is (n, x)
        img2: = image2
        label2: = label2 shape is (n, x)
        beta: = parameter of beta distributions

    Returns:
        img, label
    """
    r = np.random.beta(beta, beta)
    img = (img * r + img2 * (1 - r)).astype(np.uint8)  # from float64 to uint8
    label = np.concatenate((label, label2), axis=0)
    return img, label
