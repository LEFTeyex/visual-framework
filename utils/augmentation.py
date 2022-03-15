r"""
Data augmentation utils.
Consist of all the data augmentation methods for using datasets.
"""

import cv2
import math
import random
import numpy as np

from bbox import xywhn2xyxy

__all__ = ['random_affine_or_perspective', 'cutout', 'mixup', 'mosaic']


def random_affine_or_perspective(img, label, filter_bbox=True,
                                 perspective=0.0, angle=0.0, scale=0.0, shear=0.0, translate=0.0, flip=0.0):
    r"""
    Random affine or perspective.
    Consist of perspective, rotation, scale, shear, flip.
    Args:
        img: = image shape is (h, w, c)
        label: = label shape is (n, 5) if object detection which the format of bbox is xyxy
        filter_bbox: = True/False whether filter bbox after augment
        perspective: = (-perspective, perspective) in x and y, level is about (1e3)
        angle: = rotation angle [0, 180]
        scale: = scale in (1 - scale, 1 + scale) [0, 1]
        shear: = shear angle [0, 90]
        translate: = scaling of w and h (pixels) [0 , 1]
        flip: = the probability for x and y flip [0 , 1]

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
    angle = random.uniform(-angle, angle)  # angle to rotate
    scale = random.uniform(1 - scale, 1 + scale)  # scale ot resize
    r[:2] = cv2.getRotationMatrix2D(angle=angle, center=(w / 2, h / 2), scale=scale)  # todo center to check

    # shear
    s = np.eye(3)
    s[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear
    s[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear

    # flip
    f = np.eye(3)
    if random.uniform(0, 1) < flip:  # x flip
        f[0, [0, 2]] = -1, w - 1
    if random.uniform(0, 1) < flip:  # y flip
        f[1, [1, 2]] = -1, h - 1

    # translation
    t = np.eye(3)
    t[0, 2] = random.uniform(-translate, translate) * w  # x translation
    t[1, 2] = random.uniform(-translate, translate) * h  # y translation

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
        eps: = 1e-16

    Returns:
        filter index (ndarray bool)
    """
    w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
    w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # filter


def cutout(img, label, color_mask=False):
    r"""
    CutOut augmentation https://arxiv.org/abs/1708.04552.
    Args:
        img: = image shape is (h, w, c)
        label: = label shape is (n, x) which the format of bbox is xyxy
        color_mask: = False/True mask pixel is colorful

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

        if color_mask:
            # random color mask
            img[y_min:y_max, x_min:x_max] = [np.random.randint(64, 191, dtype=np.uint8) for _ in range(3)]
        else:
            # zero mask
            img[y_min:y_max, x_min:x_max] = np.array(0, dtype=np.uint8)

        # TODO maybe need to filter the label which is bad
        # if len(label):
        #     pass

    return img, label


def mixup(img2, label2, beta=8.0):
    r"""
    MixUp augmentation https://arxiv.org/abs/1710.09412.
    Args:
        img2: = (image1, image2) shape is (h, w, c)
        label2: = (label1, label2) shape is (n, x) which the format of bbox is xyxy
        beta: = parameter of beta distributions

    Returns:
        img2, label2
    """
    r = np.random.beta(beta, beta)
    img2 = (img2[0] * r + img2[1] * (1 - r)).astype(np.uint8)  # from float64 to uint8
    label2 = np.concatenate((label2[0], label2[1]), axis=0)
    return img2, label2


def mosaic(img4, label4, shape4, s):
    r"""
    Mosaic augmentation https://
    Args:
        img4: = (image1, ...) shape is (h, w, c)
        label4: = (label1, ...) shape is (n, x) which the format of bbox is xywhn
        shape4: = ((h, w), ...)
        s: = the size of mosaic image side

    Returns:
        img_out, label_out
    """
    label4 = np.copy(label4) if isinstance(label4, np.ndarray) else label4.clone()
    s_s = s * 2
    img_out = np.zeros((s_s, s_s, img4[0].shape[2]), dtype=np.uint8)
    label_out = []
    yc, xc = (int(random.uniform(s // 2, s + s // 2)) for _ in range(2))
    for index, (img, label, shape) in enumerate(zip(img4, label4, shape4)):
        # make mosaic image
        h, w = shape
        x1m, y1m, x2m, y2m = (0,) * 4
        x1, y1, x2, y2 = (0,) * 4
        if index == 0:  # top left
            x1m, y1m, x2m, y2m = max(xc - w, 0), max(yc - h, 0), xc, yc  # mosaic image
            x1, y1, x2, y2 = w - (x2m - x1m), h - (y2m - y1m), w, h  # image for mosaic

        elif index == 1:  # top right
            x1m, y1m, x2m, y2m = xc, max(yc - h, 0), min(xc + w, s_s), yc
            x1, y1, x2, y2 = 0, h - (y2m - y1m), min(w, x2m - x1m), h

        elif index == 2:  # bottom left
            x1m, y1m, x2m, y2m = max(xc - w, 0), yc, xc, min(yc + h, s_s)
            x1, y1, x2, y2 = w - (x2m - x1m), 0, w, min(h, y2m - y1m)

        elif index == 3:  # bottom right
            x1m, y1m, x2m, y2m = xc, yc, min(xc + w, s_s), min(yc + h, s_s)
            x1, y1, x2, y2 = 0, 0, min(w, x2m - x1m), min(h, y2m - y1m)

        img_out[x1m:x2m, y1m:y2m] = img[x1:x2, y1:y2]
        pxy = x1m - x1, y1m - y1  # for transforming label

        # transform label
        if label.size:
            label[:, 1:] = xywhn2xyxy(label[:, 1:], shape, pxy)
        label_out.append(label)

    label_out = np.concatenate(label_out, axis=0)
    label_out[:, 1:].clip(0, s_s)  # clip xyxy in mosaic image
    return img_out, label_out


if __name__ == '__main__':
    pass
