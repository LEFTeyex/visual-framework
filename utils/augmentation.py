r"""
Data augmentation utils.
Consist of all the data augmentation methods for using datasets.
"""
import random

import numpy as np

__all__ = ['cutout', 'mixup']


def cutout(img, label):
    r"""
    CutOut augmentation https://arxiv.org/abs/1708.04552.
    Args:
        img:  = image shape is (h, w, c)
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
        print(np.array(0, dtype=np.uint8))
        img[y_min:y_max, x_min:x_max] = np.array(0, dtype=np.uint8)

        # random color mask
        # img[y_min:y_max, x_min:x_max] = [np.random.randint(64, 191, dtype=np.uint8) for _ in range(3)]

        # TODO maybe need to filter the label which is bad
        # if len(label):
        #     pass

    return img, label


def mixup(img, label, img2, label2, beta=8.0):
    # MixUp augmentation https://arxiv.org/abs/1710.09412
    r = np.random.beta(beta, beta)
    img = (img * r + img2 * (1 - r)).astype(np.uint8)  # from float64 to uint8
    label = np.concatenate((label, label2), axis=0)
    return img, label
