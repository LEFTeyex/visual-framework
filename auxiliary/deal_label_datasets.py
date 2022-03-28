r"""
Consist of the functions for dealing labels for training.
1.Deal labels from xml to txt_yolo.
2.Classify datasets as train/val/test datasets.
"""
import json
import random
import argparse
import xml.etree.ElementTree as ElementTree

from tqdm import tqdm
from pathlib import Path
from pycocotools.coco import COCO
from typing import Optional

from utils.general import save_all_txt, load_all_txt

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


def _xyxy2x1y1wh(bbox):
    x1, y1 = bbox[0] - 1, bbox[1] - 1
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return [x1, y1, w, h]


def _x1y1wh2xywhn(bbox, w: int, h: int):
    dw = 1. / w
    dh = 1. / h
    x = bbox[0] + bbox[2] / 2
    y = bbox[1] + bbox[3] / 2
    w, h = bbox[2:]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


def _xyxy2segmentation(bbox):
    x1, y1, x2, y2 = bbox
    # left_top left_bottom right_bottom right_top
    segmentation = [[x1, y1, x1, y2, x2, y2, x2, y1]]
    return segmentation


def _xml2txt_yolo(path: str, classes, w: _int_or_None = None, h: _int_or_None = None,
                  filter_difficult: bool = True):
    r"""For object detection"""
    path = Path(path)
    if not path.exists():
        raise FileExistsError(f'The path {str(path)} do not exist')
    save_parent = path.parent / f'{path.name}_yolo'
    save_parent.mkdir(parents=True, exist_ok=True)
    with tqdm(path.glob('*.xml'), bar_format='{l_bar}{bar:20}{r_bar}',
              desc=f'{path.name} xml2txt_yolo', total=len(list(path.glob('*.xml')))) as pbar:
        for p in pbar:
            save_path = save_parent / f'{p.stem}.txt'
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
                difficult = obj.find('difficult')
                if difficult is None:
                    difficult = 0
                else:
                    difficult = int(difficult.text)
                # the class is not for using or it is difficult
                condition = (cls not in classes or difficult == 1) if filter_difficult else (cls not in classes)
                if condition:
                    continue
                cls_idx = [classes.index(cls)]
                bbox = obj.find('bndbox')
                bbox = [float(bbox.find(coord).text) for coord in ('xmin', 'ymin', 'xmax', 'ymax')]
                # check whether cross the border
                if bbox[0] < 1:
                    bbox[0] = 1
                if bbox[1] < 1:
                    bbox[1] = 1
                if bbox[2] > w:
                    bbox[2] = w
                if bbox[3] > h:
                    bbox[3] = h
                bbox = _xyxy2xywhn(bbox, w, h)
                label = cls_idx + bbox
                labels.append(label)
            save_all_txt((labels, save_path))
    print('Done')


def _xml2json_coco(xml_dir_path: str, image_path_txt: str, classes, supercategory: str, filter_difficult: bool = True):
    r"""Convert xml corresponding to image path txt file while it is corresponding to val or test"""
    xml_dir_path = Path(xml_dir_path)
    image_path_txt = Path(image_path_txt)
    save_path = str(image_path_txt.parent / f'{str(image_path_txt.stem)}.json')
    image_paths_list = load_all_txt(image_path_txt)

    assert xml_dir_path.exists(), f'The path {str(xml_dir_path)} do not exist'
    assert xml_dir_path.is_dir(), f'The path {str(xml_dir_path)} is not a dir'

    images = []  # consist of dict(file_name, height, width, id)
    annotations = []  # consist of dict(segmentation, area, iscrowd, image_id, bbox:list, category_id, id)
    categories = []  # # consist of dict(supercategory, id, name)
    image_id = 0
    ann_id = 0

    with tqdm(image_paths_list, bar_format='{l_bar}{bar:20}{r_bar}',
              desc=f'{image_path_txt.stem} xml2json_coco', total=len(image_paths_list)) as pbar:
        for p in pbar:
            xml_path = xml_dir_path / f'{Path(p).stem}.xml'
            root = ElementTree.parse(xml_path).getroot()
            image_name = str(p)

            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            # annotations item
            cats = []
            for obj in root.iter('object'):
                cls = obj.find('name').text
                category_id = classes.index(cls)
                cat = {'supercategory': str(supercategory),
                       'id': int(category_id),
                       'name': str(cls)}
                cats.append(cat)

                difficult = obj.find('difficult')
                if difficult is None:
                    difficult = 0
                else:
                    difficult = int(difficult.text)

                # the class is not for using or it is difficult
                condition = (cls not in classes or difficult == 1) if filter_difficult else (cls not in classes)
                if condition:
                    continue
                bbox = obj.find('bndbox')
                bbox = [float(bbox.find(coord).text) for coord in ('xmin', 'ymin', 'xmax', 'ymax')]

                # check whether cross the border
                if bbox[0] < 1:
                    bbox[0] = 1
                if bbox[1] < 1:
                    bbox[1] = 1
                if bbox[2] > w:
                    bbox[2] = w
                if bbox[3] > h:
                    bbox[3] = h
                segmentation = _xyxy2segmentation(bbox)
                bbox = _xyxy2x1y1wh(bbox)
                area = float(bbox[2] * bbox[3])
                iscrowd = 0 if difficult == 0 else 1

                ann = {'id': ann_id,
                       'image_id': image_id,
                       'category_id': category_id,
                       'segmentation': segmentation,
                       'area': area,
                       'bbox': bbox,
                       'iscrowd': iscrowd}
                annotations.append(ann)
                ann_id += 1

            # images item
            image = {'file_name': image_name,
                     'height': h,
                     'width': w,
                     'id': image_id}
            images.append(image)
            image_id += 1

            # categories item
            for cat_dict in cats:
                if cat_dict not in categories:
                    categories.append(cat_dict)

    categories.sort(key=lambda x: int(x['id']))
    coco_json = {'images': images,
                 'annotations': annotations,
                 'categories': categories}
    with open(save_path, 'w') as f:
        json.dump(coco_json, f)
    print('Done')


def _classify_datasets(path: str, save_path: str, seed: int, weights=(0.8, 0.1)):
    r"""Get absolute path for training images"""
    classify_to = ('train', 'val', 'test')
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # set seed and save
    random.seed(seed)
    save_all_txt(([seed], save_path / 'seed.txt'))

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


def convert_xml2txt_yolo_or_xml2json_coco_or_classify_datasets(args):
    kind = args.kind.lower().replace(' ', '')

    if kind == 'xml2txt' or kind == 'xml2coco':
        path = args.path_parent
        classes = args.classes
        filter_difficult = args.filter_difficult

        if kind == 'xml2txt':
            dir_xml = args.dir_xml
            w, h = args.wh
            for name in dir_xml:
                _xml2txt_yolo(Path(path) / name, classes, w, h, filter_difficult)
        else:
            image_path_txt = args.image_path_txt
            supercategory = args.supercategory
            _xml2json_coco(path, image_path_txt, classes, supercategory, filter_difficult)

    elif kind == 'classify':
        path = args.path_classify
        save_path = Path(args.save_path) / args.dir_name
        seed = args.seed
        weights = args.weights
        _classify_datasets(path, save_path, seed, weights)

    else:
        raise ValueError(f'The input kind {args.kind} is wrong')


def add_prefix_suffix_for_path_txt(list_str: list, prefix: str, suffix: str):
    for idx, x in enumerate(list_str):
        list_str[idx] = [str(Path(prefix) / f'{x}{suffix}')]
    return list_str


def deal_voc_detection(path: str, filter_difficult: bool = True):
    r"""
    Deal VOC datasets to yolo(from xml to txt).
    Path like ../VOC2012
    The dir structure is ../VOC2012|Annotations|2007_000027.xml
                                   |           |    ...
                                   |
                                   |JPEGImages|2007_000027.jpg
                                   |          |    ...
                                   |
                                   |ImageSets|Main|trainval.txt
                                   |              |train.txt
                                   |              |val.txt
                                   |              | ...
                                   |  ...
    """
    path = Path(path)
    if not path.exists():
        raise FileExistsError(f'The path {str(path)} do not exist')

    # convert labels
    image_dir = path / 'JPEGImages'
    label_dir = path / 'Annotations'
    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    _xml2txt_yolo(str(label_dir), classes, filter_difficult=filter_difficult)
    label_dir = path / 'Annotations_yolo'

    # deal train and val txt
    deal_txt = ['trainval.txt', 'train.txt', 'val.txt', 'test.txt']
    save_txt_path = [path / txt for txt in deal_txt]
    deal_txt = [str(path / 'ImageSets/Main' / txt) for txt in deal_txt]
    prefix = str(path / 'images')
    suffix = '.jpg'

    # check and filter deal_txt exists
    deal_txt = [x for x in deal_txt if Path(x).exists()]

    all_txt_list = load_all_txt(*deal_txt)
    for idx, txt_list in enumerate(all_txt_list):
        all_txt_list[idx] = add_prefix_suffix_for_path_txt(txt_list, prefix, suffix)
    save_all_txt(*zip(all_txt_list, save_txt_path))

    # reset name for JPEGImages and Annotations_yolo
    image_dir.rename(path / 'images')
    label_dir.rename(path / 'labels')
    print('Done')


def deal_coco_detection(path: str):
    r"""
    Deal COCO datasets to yolo(from json to txt).
    Path like ../COCO2017
    The dir structure is ../COCO2017|annotations|instances_train2017.json
                                    |           |instances_val2017.json
                                    |           |       ...
                                    |
                                    |
                                    |images|train2017
                                           |val2017
                                           |  ...
    """
    path = Path(path)
    if not path.exists():
        raise FileExistsError(f'The path {str(path)} do not exist')
    image_dir = path / 'images'
    label_dir = path / 'annotations'
    deal_json = ['instances_train2017.json', 'instances_val2017.json']
    for json_path in deal_json:
        # get data of image from json file by pycocotools.coco.COCO
        data = COCO(label_dir / json_path)
        data_type = json_path[10:-5]
        img_path_save_path = path / f'{data_type}.txt'
        cats_dict = {str(d['id']): str(k) for k, d in data.cats.items()}

        with tqdm(data.imgToAnns.items(), bar_format='{l_bar}{bar:20}{r_bar}',
                  desc=f'{data_type} json2txt_yolo', total=len(data.imgToAnns.items())) as pbar:
            # deal annotations to labels
            for Id, annotations in pbar:
                img_data = data.imgs[Id]
                img_name = img_data['file_name']
                img_path = [str(image_dir / data_type / img_name)]
                label_name = img_name.replace('.jpg', '.txt')
                h, w = img_data['height'], img_data['width']
                label_save_path = path / f"labels/{data_type}/{label_name}"
                label_save_path.parent.mkdir(parents=True, exist_ok=True)
                labels = []
                for ann in annotations:
                    bbox = ann['bbox']
                    cls_idx = cats_dict[str(ann['category_id'])]
                    bbox = _x1y1wh2xywhn(bbox, w, h)
                    label = cls_idx + bbox
                    labels.append(label)
                save_all_txt((labels, label_save_path))
                # notice that if you run this function it will write repeat path in *.txt
                save_all_txt((img_path, img_path_save_path), mode='a')
    print('Done')


def parse_args_detect(known: bool = False):
    r"""
    Parse args for training.
    Args:
        known: bool = True or False, Default=False
            parser will get two namespace which the second is unknown args, if known=True.

    Return namespace(for setting args)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', type=str, default='xml2coco', help='xml2coco / xml2txt / classify')

    # xml2txt_yolo and xml2coco_json
    parser.add_argument('--path_parent', type=str,
                        default=r'F:\datasets\good\VOCdevkit\VOC2012\Annotations', help='')
    parser.add_argument('--filter_difficult', type=bool, default=True, help='')
    # parser.add_argument('--classes', type=list,
    #                     default=['holothurian', 'echinus', 'scallop', 'starfish'],
    #                     help='URPC')
    parser.add_argument('--classes', type=list,
                        default=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                                 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                                 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
                        help='VOC')
    # parser.add_argument('--classes', type=list,
    #                     default=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    #                              'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    #                              'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    #                              'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    #                              'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    #                              'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
    #                              'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    #                              'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    #                              'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    #                              'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    #                              'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'],
    #                     help='COCO')

    # xml2coco_json
    parser.add_argument('--image_path_txt', type=str,
                        default=r'F:\datasets\good\VOCdevkit\VOC2012\val.txt', help='path')
    parser.add_argument('--supercategory', type=str, default='voc', help='')

    # xml2txt_yolo
    parser.add_argument('--dir_xml', type=list, default=['box'], help='')
    parser.add_argument('--wh', type=tuple, default=(None, None), help='')

    # classify_datasets
    parser.add_argument('--path_classify', type=str, default='F:/datasets/VOCdevkit/VOC2012/images', help='')
    parser.add_argument('--save_path', type=str, default=str(ROOT.parent / 'data/datasets'), help='')
    parser.add_argument('--dir_name', type=str, default='VOC2012', help='')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--weights', type=list, default=[0.8, 0.1],
                        help='the proportion of train and val, the rest is test')
    namespace = parser.parse_known_args()[0] if known else parser.parse_args()
    return namespace


def main():
    args = parse_args_detect(False)
    convert_xml2txt_yolo_or_xml2json_coco_or_classify_datasets(args)


def check_json_coco():
    path = 'F:/datasets/good/VOCdevkit/VOC2012/trainval.json'  # json path
    data = COCO(path)
    pass


if __name__ == '__main__':
    main()
    # check_json_coco()
    # deal_voc_detection()
    # deal_coco_detection()
