r"""
Custom Dataset module.
Consist of Dataset that designed for different tasks.
"""

import cv2
import json
import torch
import random
import numpy as np

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset

from utils.log import LOGGER
from utils.bbox import xywhn2xyxy, xyxy2xywhn, rescale_xywhn
from utils.general import load_all_yaml, to_tuplex, make_divisible_up
from utils.augmentation import random_affine_or_perspective, cutout, mixup, mosaic
from utils.typeslib import _path, _int_or_tuple

__all__ = ['DatasetDetect', 'get_and_check_datasets_yaml', 'load_image_resize', 'load_image_correct_rect']

IMAGE_FORMATS = ('bmp', 'jpg', 'jpeg', 'jpe', 'png', 'tif', 'tiff', 'webp')  # acceptable image suffixes


class DatasetDetect(Dataset):
    r"""
    Load images and labels while check them for detection.
    For images, save its paths and load images with resizing, padding.
    For labels, save it and convert labels when images resized or padded.
    Args:
        datasets: = datasets dict
        name: = the name of dataset type
        img_size: int = the largest edge of image size
        augment: = False/True whether data augment
        data_augment: str = 'cutout'/'mixup'/'mosaic' the kind of data augmentation
        hyp: = self.hyp in train hyperparameter augmentation
        json_gt: = path for saving json whether create json coco format for using COCOeval
    """

    # TODO upgrade val_from_train, test_from_train function in the future reference to random_split
    # TODO upgrade rectangular train shape for image model in the future reference to yolov5 rect=True
    # TODO upgrade .cache file in memory for faster training
    def __init__(self, datasets, name: str, img_size: int, augment=False, data_augment: str = '', hyp=None,
                 json_gt=None):
        self.datasets = datasets
        self.img_size = img_size
        self.augment = augment
        self.data_augment = data_augment.lower().replace(' ', '')
        self.hyp = hyp
        self.json_gt = json_gt
        self.img_files = get_img_files(datasets[name])  # get the path tuple of image files
        # check img suffix and img_files(to str)
        self.img_files = [str(x) for x in self.img_files if (x.suffix.replace('.', '').lower() in IMAGE_FORMATS)]
        self.img_files = tuple(self.img_files)  # to save memory
        if not self.img_files:
            raise FileNotFoundError('No images found')
        self.indices = range(len(self.img_files))  # for choices in data augmentation and coco evaluation

        # get label_files that it is str and sorted with images_files
        self.label_files = img2label_files(self.img_files)
        # check images and labels then get labels which is [np.array shape(n,nlabel), ...]
        self.label_files, self.nlabel = \
            self._check_img_get_label_create_json_coco_detect(self.img_files, self.label_files, name)
        LOGGER.info(f'Load {len(self.img_files)} images and {len(self.label_files)} labels')

    def __getitem__(self, index):
        # TODO upgrade a save list for first training to improve the speed up later training

        if self.data_augment == 'mosaic':
            image, label, shape_convert = self.load_mosaic(index)
        elif self.data_augment == 'mixup':
            image, label, shape_convert = self.load_mixup(index)
        elif self.data_augment == 'cutout':
            image, label, shape_convert = self.load_cutout(index)
        else:
            image, label, shape_convert = self.load_without_data_augment(index)

        # deal image
        image = np.transpose(image, (2, 0, 1))  # (h,w,c) to (c,h,w)
        image = np.ascontiguousarray(image)  # make image contiguous in memory
        image = torch.from_numpy(image)  # image to tensor

        # deal label
        nl = len(label)
        if nl:
            index_label = torch.zeros((nl, 1))
            label = torch.from_numpy(label)  # label to tensor
            label = torch.cat((index_label, label), dim=1)  # label shape is (nl, 1 + nlabel)
        else:
            label = torch.empty((0, self.nlabel + 1))  # empty

        return image, label, shape_convert, index  # only one image(c,h,w), label(nl, 1 + nlabel)

    def __len__(self):
        r"""Return len of all data"""
        return len(self.img_files)

    def load_without_data_augment(self, index):
        # load image path and label
        img_path = self.img_files[index]  # path str
        label = self.label_files[index].copy()  # label ndarray [[class,x,y,w,h], ...] when nlabel=5
        nl = len(label)  # number of label

        # load image and resize it
        # TODO maybe the shape computed in __init__ and save will be better and faster
        image, hw0, hw_nopad, ratio = load_image_resize(img_path, self.img_size)
        image, hw_pad, padxy = letterbox(image, self.img_size)  # pad image to shape or img_size
        shape_convert = hw0, ratio, padxy  # for convert coordinate from hw_pad to hw0 in COCOeval

        # convert label xywh normalized to xyxy (padded)
        if nl:
            label[:, 1:] = xywhn2xyxy(label[:, 1:], hw_nopad, padxy)

        if self.augment:
            image, label = random_affine_or_perspective(image, label, self.hyp['filter_bbox'],
                                                        perspective=self.hyp['perspective'],
                                                        angle=self.hyp['angle'],
                                                        scale=self.hyp['scale'],
                                                        shear=self.hyp['shear'],
                                                        translate=self.hyp['translate'],
                                                        flip=self.hyp['flip'])
        # convert label xyxy to xywh normalized (hw_pad)
        if nl:
            label[:, 1:] = xyxy2xywhn(label[:, 1:], hw_pad)

        return image, label, shape_convert

    def load_cutout(self, index):
        label = self.label_files[index].copy()
        img_path = self.img_files[index]
        image, _, hw_nopad, _ = load_image_resize(img_path, self.img_size)
        image, label = cutout(image, label)
        image, hw_pad, padxy = letterbox(image, self.img_size)
        if len(label):
            label[:, 1:] = rescale_xywhn(label[:, 1:], hw_nopad, hw_pad, padxy)
        return image, label, None  # shape_convert

    def load_mixup(self, index):
        image2, label2 = [], []
        indices = [index] + [random.choice(self.indices)]
        for idx in indices:
            label = self.label_files[idx].copy()
            img_path = self.img_files[idx]
            image, _, hw_nopad, _ = load_image_resize(img_path, self.img_size)
            image, hw_pad, padxy = letterbox(image, self.img_size)
            if len(label):
                label[:, 1:] = rescale_xywhn(label[:, 1:], hw_nopad, hw_pad, padxy)
            image2.append(image)
            label2.append(label)

        image2, label2 = mixup(image2, label2)
        return image2, label2, None  # shape_convert

    def load_mosaic(self, index):
        image4, label4, shape4 = [], [], []
        indices = [index] + random.choices(self.indices, k=3)
        for idx in indices:
            label = self.label_files[idx].copy()
            img_path = self.img_files[idx]
            image, _, hw_nopad, _ = load_image_resize(img_path, self.img_size)
            image4.append(image)
            label4.append(label)
            shape4.append(hw_nopad)

        image4, label4 = mosaic(image4, label4, shape4, self.img_size)
        return image4, label4, None  # shape_convert

    @staticmethod
    def collate_fn(batch):
        r"""
        For Dataloader batch data at last.
        Args:
            batch: list = [0(image, label, ...), 1, 2, 3] for batch index of data

        Return images, labels, shape_converts, img_ids
        """
        # batch is [(image, label, shape_convert), ...]
        images, labels, shape_converts, img_ids = zip(*batch)  # every element is tuple
        # add index for label to image
        for index, label in enumerate(labels):
            label[:, 0] = index

        labels = torch.cat(labels, dim=0)  # shape (n, 1 + nlabel)
        images = torch.stack(images, dim=0)  # shape (bs, c, h, w)
        # shape_converts is tuple
        return images, labels, shape_converts, img_ids

    def _check_img_get_label_create_json_coco_detect(self, image_files, label_files, prefix: str = '',
                                                     nlabel: int = 5):
        r"""
        Check image (channels) and labels (empty, shape, positive, normalized).
        Get labels then.
        Args:
            image_files: = images_files(path tuple)
            label_files: = labels_files(path tuple)
            prefix: str = the prefix for tqdm.tqdm
            nlabel: int = number for detecting (x,y,w,h,object) new

        Return labels(tuple), nlabel
        """
        labels = []  # save labels
        channel = cv2.imread(image_files[0]).shape[-1]
        space = ' ' * 11

        # for json coco format
        images = []  # consist of dict(file_name, height, width, id)
        annotations = []  # consist of dict(segmentation, area, iscrowd, image_id, bbox:list, category_id, id)
        categories = []  # # consist of dict(supercategory, id, name)
        ann_id = 0

        def xywh2segmentation(b):
            bx1, by1, bw, bh = b
            bx2, by2 = bx1 + bw, by1 + bh
            return [[bx1, by1, bx1, by2, bx2, by2, bx2, by1]]

        with tqdm(enumerate(zip(image_files, label_files)),
                  bar_format='{l_bar}{bar:20}{r_bar}',
                  desc=f'{space}{prefix}: checking image and label',
                  total=len(image_files)) as pbar:
            for image_id, (ip, lp) in pbar:  # image path, label path
                image = cv2.imread(ip)
                height, width, c = image.shape

                # check image
                assert c == channel, f'The channel of the image {ip} do not match with {channel}'

                # create images the image_id is corresponding to the index of image_files
                if self.json_gt:
                    image = {'file_name': Path(ip).name,
                             'height': height,
                             'width': width,
                             'id': image_id}
                    images.append(image)

                # check label and get read it to list
                with open(lp, 'r') as f:
                    label = [x.strip().split() for x in f.read().splitlines() if x]  # label is [[class,x,y,w,h], ...]
                    label = np.array(label, dtype=np.float32)
                    label = np.unique(label, axis=0)  # remove the same one
                    assert len(label), f'The label {lp} is empty'  # make sure the label is not empty
                    assert label.ndim == 2, f'There are format problem with label {lp}'
                    assert label.shape[1] == nlabel, f'The label require {nlabel} element {lp}'
                    assert (label >= 0).all(), f'The value in label should not be negative {lp}'
                    assert (label[:, 1:] <= 1).all(), f'Non-normalized or out of bounds coordinates {lp}'
                    labels.append(label)

                    # create annotations
                    if self.json_gt:
                        cls, x, y, w, h = label[:, 0], label[:, 1] * width, label[:, 2] * height, \
                                          label[:, 3] * width, label[:, 4] * height
                        # center to top-left
                        x = x - w * 0.5
                        y = y - h * 0.5
                        bboxes = np.stack((x, y, w, h), axis=0).T
                        areas = bboxes[:, 2] * bboxes[:, 3]
                        for category_id, bbox, area in zip(cls.tolist(), bboxes.tolist(), areas.tolist()):
                            bbox = [float(x) for x in bbox]
                            ann = {'id': ann_id,
                                   'image_id': image_id,
                                   'category_id': int(category_id),
                                   'segmentation': xywh2segmentation(bbox),
                                   'area': float(area),
                                   'bbox': bbox,
                                   'iscrowd': 0}
                            annotations.append(ann)
                            ann_id += 1

        # create categories and save json for coco
        if self.json_gt:
            for idx, name in enumerate(self.datasets['names']):
                cat = {'supercategory': '0',
                       'id': idx,
                       'name': name}
                categories.append(cat)
            categories.sort(key=lambda x: int(x['id']))
            coco_json = {'images': images,
                         'annotations': annotations,
                         'categories': categories}
            with open(self.json_gt, 'w') as f:
                json.dump(coco_json, f)
            LOGGER.info(f'Create json coco_gt {self.json_gt} successfully')

        return tuple(labels), nlabel


def load_image_resize(img_path: _path, img_size: int):
    r"""
    Load image and resize it which the largest edge is img_size.
    Args:
        img_path: _path = Path
        img_size: int = img_size for the largest edge

    Returns:
        image, (h0, w0), (h1, w1), r
    """
    image = cv2.imread(img_path)  # (h,w,c) BGR
    image = image[..., ::-1]  # BGR to RGB
    if image is None:
        raise FileNotFoundError(
            f'The image is None, path error or image error {img_path}')

    # TODO the process(or rect) may be faster for training in __init__ in the future
    h0, w0 = image.shape[:2]  # original hw
    r = img_size / max(h0, w0)  # ratio for resize
    h1, w1 = round(h0 * r), round(w0 * r)

    if r != 1:
        # todo args can change
        image = cv2.resize(image, dsize=(w1, h1),  # conv2.resize dsize need (w, h)
                           interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    return image, (h0, w0), (h1, w1), r


def load_image_correct_rect(img_path: _path, img_size: int, divisor: int = 32):
    r"""Load image and resize it and make its side is divisible 32"""
    image, (h0, w0), (h1, w1), r = load_image_resize(img_path, img_size)

    # compute shape_pad
    if h1 != w1:
        h2, w2 = (make_divisible_up(h1, divisor), w1) if h1 < w1 else (h1, make_divisible_up(w1, divisor))
        image, (h2, w2), padxy = letterbox(image, (h2, w2))
    else:
        h2, w2 = h1, w1
        padxy = (0, 0)

    return image, (h0, w0), (h1, w1), r, (h2, w2), padxy


def letterbox(image: np.ndarray, shape_pad: _int_or_tuple, color: tuple = (0, 0, 0)):
    r"""
    Pad image to specified shape.
    Args:
        image: np.ndarray = image(ndarray)
        shape_pad: _int_or_tuple = (h, w) or int
        color: tuple = RGB

    Returns:
        image(ndarray), (h2, w2), pxy
    """
    if isinstance(shape_pad, int):
        shape_pad = to_tuplex(shape_pad, 2)
    shape = image.shape[:2]  # current shape (h,w)
    assert shape[0] <= shape_pad[0] and shape[1] <= shape_pad[1], \
        f'The image shape: {shape} must be less than shape_pad: {shape_pad}'

    # get dh, dw for padding
    dh, dw = shape_pad[0] - shape[0], shape_pad[1] - shape[1]
    dh /= 2  # divide padding into 2 sides
    dw /= 2

    # maybe top left paddings will be smaller than bottom right
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    # add border(pad)
    image = cv2.copyMakeBorder(image, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=color)
    h2, w2 = image.shape[:2]
    padxy = (left, top)  # for converting labels as padxy
    return image, (h2, w2), padxy


def get_img_files(path, file_path_is_absolute: bool = False):
    r"""
    Get all the image path in the path and its dir.
    Args:
        path: = pathlike or [pathlike, ...]
        file_path_is_absolute: bool = True/False whether the path in file is

    Returns:
        img_files(which is absolute image paths tuple)
    """
    img_files = []
    for p in path if isinstance(path, (list, tuple)) else [path]:
        p = Path(p)
        if p.is_dir():
            img_files += [x for x in p.rglob('*.*')]

        elif p.is_file():
            with open(p, 'r') as f:
                f = f.read().splitlines()
                if file_path_is_absolute:
                    # get global path directly
                    for element in f:
                        img_files.append(Path(element.strip()))
                else:
                    # local to global path from /*.jpg or *.jpg
                    parent = p.parent  # means that the file and the image to load is in the same directory
                    for element in f:
                        element = Path(element.strip())
                        if '\\' in element.parts:  # remove / in the front of it
                            element = parent / str(element)[1:]
                            img_files.append(element)
                        else:
                            img_files.append(parent / element)
        else:
            raise TypeError(f'Something wrong with {p} in the type of file')
    return tuple(img_files)  # to save memory


def img2label_files(img_files):
    r"""
    Change image path to label path from image paths.
    The file name must be 'images' and 'labels'.
    Args:
        img_files: = img_files

    Return label_files
    """
    # change 'images' to 'labels' and change suffix to '.txt'
    label_files = ['labels'.join(p.rsplit('images', 1)).rsplit('.', 1)[0] + '.txt' for p in img_files]
    return tuple(label_files)  # to save memory


def get_and_check_datasets_yaml(path: _path):
    r"""
    Get path and other data of datasets yaml for training and check datasets yaml.
    Args:
        path: _path = Path

    Return datasets
    """
    LOGGER.info('Checking and loading the datasets yaml file...')
    # check path and get them
    datasets: dict = load_all_yaml(path)
    parent_path = Path(datasets['path'])
    train, val, test = datasets['train'], datasets.get('val'), datasets.get('test')

    # deal str or list for train, val, test
    tvt = []  # to save (train, val, test) dealt
    for file in (train, val, test):
        if file is None:
            pass
        elif isinstance(file, str):
            tvt.append(parent_path / file)
        elif isinstance(file, (list, tuple)):
            save_tem = []
            for element in file:
                save_tem.append(parent_path / element)
            tvt.append(save_tem)
        else:
            raise TypeError(f'The type of {file} is wrong, '
                            f'please reset in the {path}')
    # get the value dealt
    try:
        train, val, test = tvt
    except ValueError:
        LOGGER.debug('No datasets test')
        train, val = tvt

    # train must exist
    if train is None:
        raise FileExistsError(f'The path train must not None, '
                              f'please reset it in the {path}')
    # val must exist
    if val is None:
        raise FileExistsError(f'The path val must not None, '
                              f'please reset it in the {path}')

    # check whether train, val, test exist
    for path in (train, val, test):
        for p in path if isinstance(path, list) else [path]:
            if p is None:
                pass
            elif not p.exists():
                raise FileExistsError(f'The path {p} do not exists, '
                                      f'please reset in the {p}')

    def to_str(obj):
        if isinstance(obj, list):
            obj = [str(x) for x in obj]
        else:
            obj = str(obj)
        return obj

    # convert pathlike to str type
    train = to_str(train)
    val = to_str(val)
    if test is not None:
        test = to_str(test)

    datasets['train'], datasets['val'], datasets['test'] = train, val, test

    # check number of classes and name
    if not (datasets['nc'] == len(datasets['names'])):
        raise ValueError('There is no one-to-one correspondence '
                         'between the number of classes and its names')

    LOGGER.info('Get the path for training successfully')
    return datasets


if __name__ == '__main__':
    pass
