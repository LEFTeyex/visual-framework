r"""
Consist of the functions for dealing labels for training.
1.Deal labels from xml to txt_yolo.
2.Classify datasets as train/val/test datasets.
"""

import argparse
import xml.etree.ElementTree as ElementTree

from pathlib import Path
from typing import Optional

from utils.general import save_all_txt

_int_or_None = Optional[int]


def _xyxy2xywhn(bbox, w, h):
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


def _xml2txt_yolo(path: str, classes: list, w: _int_or_None = None, h: _int_or_None = None,
                  filter_difficult: bool = True):
    r"""For object detection"""
    path = Path(path)
    if not path.exists():
        raise FileExistsError(f'The path {str(path)} do not exist')
    save_parent = path.parent / (path.name + '_yolo')
    save_parent.mkdir(parents=True, exist_ok=True)
    for p in path.glob('*.xml'):
        save_path = save_parent / (p.stem + '.txt')
        root = ElementTree.parse(p).getroot()
        size = root.find('size')
        if size is None:
            assert w and h, f'No size in {str(p)}, please input w and h for normalized xywh'
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


# TODO 2022.3.20
def _classify_datasets():
    pass


def convert_xml2txt_yolo_or_classify_datasets():
    pass


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
    parser.add_argument('--classes', type=list, default=['person'], help='')
    parser.add_argument('--names', type=list, default=['train', 'val', 'test'], help='')
    namespace = parser.parse_known_args()[0] if known else parser.parse_args()
    return namespace


if __name__ == '__main__':
    _xml2txt_yolo('../data/images', ['person'])
