r"""
Consist of the functions for dealing labels for training.
1.Deal labels from xml to txt_yolo.
2.Classify datasets as train/val/test datasets.
"""

import random
import argparse
import xml.etree.ElementTree as ElementTree

from tqdm import tqdm
from pathlib import Path
from typing import Optional

from utils.general import save_all_txt

ROOT = Path.cwd()

_int_or_None = Optional[int]


def _xyxy2xywhn(bbox, w: int, h: int):
    dw = 1. / w
    dh = 1. / h
    # the coordinate in xml is started form 1, but wanted 0 so it minus 1
    x = (bbox[0] + bbox[2]) / 2. - 1
    y = (bbox[1] + bbox[3]) / 2. - 1
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


def _xml2txt_yolo(path: str, classes, w: _int_or_None = None, h: _int_or_None = None,
                  filter_difficult: bool = True):
    r"""For object detection"""
    path = Path(path)
    if not path.exists():
        raise FileExistsError(f'The path {str(path)} do not exist')
    save_parent = path.parent / (path.name + '_yolo')
    save_parent.mkdir(parents=True, exist_ok=True)
    with tqdm(path.glob('*.xml'), bar_format='{l_bar}{bar:20}{r_bar}',
              desc=f'{path.name} xml2txt_yolo', total=len(list(path.glob('*.xml')))) as pbar:
        for p in pbar:
            save_path = save_parent / (p.stem + '.txt')
            root = ElementTree.parse(p).getroot()
            size = root.find('size')
            if size is None:
                assert isinstance(w, int) and isinstance(h, int), \
                    f'No size in {str(p)}, please input w and h for normalized xywh'
            else:
                w = int(size.find('width').text)
                h = int(size.find('height').text)

            labels = []
            for obj in root.iter('object'):
                cls = obj.find('name').text
                difficult = obj.find('difficult').text
                # the class is not for using or it is difficult
                condition = (cls not in classes or int(difficult) == 1) if filter_difficult else (cls not in classes)
                if condition:
                    continue
                cls_idx = [classes.index(cls)]
                bbox = obj.find('bndbox')
                bbox = [float(bbox.find(coord).text) for coord in ('xmin', 'ymin', 'xmax', 'ymax')]
                # check whether cross the border
                if bbox[0] < 0:
                    bbox[0] = 0
                if bbox[1] < 0:
                    bbox[1] = 0
                if bbox[2] > w:
                    bbox[2] = w
                if bbox[3] > h:
                    bbox[3] = h
                bbox = _xyxy2xywhn(bbox, w, h)
                label = cls_idx + bbox
                labels.append(label)
            save_all_txt((labels, save_path))
    print('Done')


def _classify_datasets(path: str, save_path: str, seed: int, weights=(0.8, 0.1)):
    r"""Get absolute path for training images"""
    classify_to = ('train', 'val', 'test')
    print(save_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # set seed
    random.seed(seed)
    save_all_txt(([[seed]], save_path / 'seed.txt'))

    # get image paths
    image_formats = ('bmp', 'jpg', 'jpeg', 'jpe', 'png', 'tif', 'tiff', 'webp')  # acceptable image suffixes
    img_files = []
    p = Path(path)
    if p.is_dir():
        img_files += [x for x in p.glob('*.*')]
    else:
        raise TypeError(f'The path {p} is not a directory, please input the path of directory')
    img_files = [str(x) for x in img_files if x.suffix.replace('.', '').lower() in image_formats]

    # sample
    total_num = len(img_files)
    kind_num = {}
    for kind, weight in zip(classify_to, weights):
        kind_num[kind] = round(total_num * weight)
    train_val = random.sample(range(total_num), sum(kind_num.values()))
    val = random.sample(train_val, kind_num['val'])

    # classify
    train_list, val_list, test_list = [], [], []
    with tqdm(enumerate(img_files), bar_format='{l_bar}{bar:20}{r_bar}',
              desc=f'{p.name} classify', total=total_num) as pbar:
        for idx, img_path in pbar:
            if idx in train_val:
                if idx in val:
                    val_list.append([img_path])
                else:
                    train_list.append([img_path])
            else:
                test_list.append([img_path])

    # check
    assert (len(train_list) + len(val_list) + len(test_list)) == total_num, \
        f'The total number {total_num} is not right after classifying'

    print('Saving...')
    save_all_txt((train_list, save_path / 'train.txt'),
                 (val_list, save_path / 'val.txt'),
                 (test_list, save_path / 'test.txt'))
    print('Done')


def convert_xml2txt_yolo_or_classify_datasets(args):
    kind = args.kind.lower().replace(' ', '')
    if kind == 'xml2txt':
        path = args.path_parent
        dir_xml = args.dir_xml
        w, h = args.wh
        classes = args.classes
        filter_difficult = args.filter_difficult
        for name in dir_xml:
            _xml2txt_yolo(Path(path) / name, classes, w, h, filter_difficult)

    elif kind == 'classify':
        path = args.path_classify
        save_path = args.save_path
        seed = args.seed
        weights = args.weights
        _classify_datasets(path, save_path, seed, weights)

    else:
        raise ValueError(f'The input kind {args.kind} is wrong')


def parse_args_detect(known: bool = False):
    r"""
    Parse args for training.
    Args:
        known: bool = True or False, Default=False
            parser will get two namespace which the second is unknown args, if known=True.

    Return namespace(for setting args)
    """
    _str_or_None = Optional[str]
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', type=str, default='classify', help='xml2txt / classify')
    # xml2txt_yolo
    parser.add_argument('--path_parent', type=str, default='F:/datasets/VOCdevkit/VOC2012', help='')
    parser.add_argument('--dir_xml', type=list, default=['Annotations'], help='')
    parser.add_argument('--classes', type=list,
                        default=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                                 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                                 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
                        help='')
    parser.add_argument('--wh', type=tuple, default=(None, None), help='')
    parser.add_argument('--filter_difficult', type=bool, default=False, help='')
    # classify_datasets
    parser.add_argument('--path_classify', type=str, default='F:/datasets/VOCdevkit/VOC2012/JPEGImages', help='')
    parser.add_argument('--save_path', type=str, default=(ROOT.parent / 'data/datasets/data1'), help='')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--weights', type=list, default=[0.8, 0.1],
                        help='the proportion of train and val, the rest is test')
    namespace = parser.parse_known_args()[0] if known else parser.parse_args()
    return namespace


def main():
    args = parse_args_detect(False)
    convert_xml2txt_yolo_or_classify_datasets(args)


if __name__ == '__main__':
    main()