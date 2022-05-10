r"""
For validating model.
Consist of some Valers.
"""

import json
import torch
import argparse

from pathlib import Path

from ..utils import WRITER
from ..utils.log import LOGGER, add_log_file, log_results
from ..utils.loss import LossDetectYolov5
from ..utils.bbox import rescale_xyxy, xyxy2x1y1wh
from ..utils.datasets import get_and_check_datasets_yaml, DatasetDetect
from ..utils.general import timer, select_one_device, save_all_yaml, load_all_yaml, loss_to_mean, hasattr_not_none
from ..metaclass.metavaler import MetaValDetect
from ..models.yolov5.yolov5_v6 import yolov5s_v6

__all__ = ['ValDetect']

ROOT = Path.cwd()


# TODO whether the different batch size comparing to training will affect the val mAP

class _Args(object):
    def __init__(self, args):
        self.inc = args.inc
        self.hyp = args.hyp
        self.task = args.task
        self.half = args.half
        self.device = args.device
        self.weights = args.weights
        self.workers = args.workers
        self.datasets = args.datasets
        self.save_name = args.save_name
        self.save_path = args.save_path
        self.image_size = args.image_size
        self.batch_size = args.batch_size
        self.pin_memory = args.pin_memory
        self._load_model = args.load_model


class ValDetect(_Args, MetaValDetect):
    def __init__(self, args=None, model=None, writer=None, half=True, dataloader=None, loss_fn=None,
                 cls_names=None, epoch=-1, visual_image=None, coco_json=None, hyp=None):
        # all need
        self.epoch = epoch
        self.writer = writer
        self.training = model is not None

        if self.training:
            self.hyp = hyp
            self.half = half
            self.loss_fn = loss_fn
            self.cls_names = cls_names
            self.coco_json = coco_json
            self.dataloader = dataloader
            self.visual_image = visual_image
            self.device = next(model.parameters()).device

            if self.half and self.device.type == 'cpu':
                LOGGER.warning(f'The device is {self.device}, half precision only supported on CUDA')
                self.half = False
            self.model = model.half() if self.half else model.float()

        else:
            super(ValDetect, self).__init__(args)
            self.path_dict = self.set_save_path(('hyp', 'hyp.yaml'), ('logger', 'logger.log'), ('args', 'args.yaml'),
                                                ('datasets', 'datasets.yaml'), ('json_gt', 'json_gt.json'),
                                                ('json_dt', 'json_dt.json'), ('coco_results', 'coco_results.json'))
            # Add FileHandler for logger
            add_log_file(self.path_dict['logger'])

            self.coco_json = {
                'test': self.path_dict['json_gt'],
                'dt': self.path_dict['json_dt']
            }

            self.device = select_one_device(self.device)
            self.datasets = get_and_check_datasets_yaml(self.datasets)

            self.hyp = load_all_yaml(self.hyp)
            nl = int(len(self.datasets['anchors']))
            self.hyp['bbox'] *= 3 / nl
            self.hyp['cls'] *= self.datasets['nc'] / 80 * 3. / nl
            self.hyp['obj'] *= (self.image_size / 640) ** 2 * 3. / nl

            self.cls_names = self.datasets['names']

            save_all_yaml(
                (self.hyp, self.path_dict['hyp']),
                (vars(args), self.path_dict['args']),
                (self.datasets, self.path_dict['datasets'])
            )

            self.checkpoint = self.load_checkpoint(self.weights)

            self.model = self.load_model(
                yolov5s_v6(self.inc, self.datasets['nc'], self.datasets['anchors'], self.image_size),
                **self._load_model
            )
            self.model = model.half() if self.half else model.float()

            self.loss_fn = LossDetectYolov5(self.model, self.hyp)
            self.dataloader = self.set_dataloader(
                DatasetDetect(self.datasets, self.task, self.image_size, coco_gt=self.coco_json['test']))

    @torch.no_grad()
    def val_training(self):
        self.model.eval()
        data_dict = self.val_once(('total', 'bbox', 'class', 'object'))
        json_dt = data_dict['json_dt']
        img_ids = self.dataloader.dataset.indices

        if self.epoch != -1:
            coco_eval, coco_stats = self.coco_evaluate(self.coco_json['val'], json_dt, img_ids, 'bbox')
            self._log_writer(coco_stats, writer=True)
        else:
            coco_eval, coco_stats = self.coco_evaluate(self.coco_json['test'], json_dt, img_ids, 'bbox',
                                                       print_result=True)
            self._log_writer(coco_stats)
        if self.half:
            self.model.float()
        return coco_eval, coco_stats

    # @torch.inference_mode()
    @torch.no_grad()
    def val(self):
        self.model.eval()
        data_dict = self.val_once(('total', 'bbox', 'class', 'object'))
        json_dt = data_dict['json_dt']
        img_ids = self.dataloader.dataset.indices
        coco_eval, coco_stats = self.coco_evaluate(self.coco_json['test'], json_dt, img_ids, 'bbox', print_result=True)
        self._log_writer(coco_stats)
        self.save_coco_results(coco_eval, self.path_dict['coco_results'])
        self.release_cuda_cache()

    def _log_writer(self, coco_stats, writer=False):
        LOGGER.debug(f'{[v.tolist() for v in coco_stats]}')
        result_names = ('AP5095', 'AP50', 'AP75')
        results = [v.tolist() for v in coco_stats[:3]]
        log_results(results, result_names)
        if writer:
            WRITER.add_epoch_curve(self.writer, 'val_metrics', results, result_names, self.epoch)

    def visual_dataset(self, dataset, name):
        pass

    def preprocess(self, data_dict):
        self.model.eval()
        data_dict['json_dt'] = []
        return data_dict

    def preprocess_iter(self, data, data_dict):
        # data (images, labels, shape_converts, img_ids)
        x, labels, shape_converts, img_ids = data
        data_dict['shape_converts'] = shape_converts
        data_dict['img_ids'] = img_ids
        x = x.to(self.device)
        labels = labels.to(self.device)
        x = (x.half() / 255) if self.half else (x.float() / 255)
        data_dict['x'] = x
        return x, labels, None, data_dict

    @staticmethod
    def mean_loss(index, loss_mean, loss, data_dict):
        (loss_items,) = data_dict['other_loss_iter']
        loss = torch.cat((loss.detach(), loss_items.detach()))
        loss_mean = loss_to_mean(index, loss_mean, loss)
        return loss_mean, data_dict

    def decode_iter(self, outputs, data_dict):
        # parse outputs to predictions bbox is xyxy
        outputs = self.model.decode(outputs,
                                    self.hyp['obj_threshold'],
                                    self.hyp['iou_threshold'],
                                    self.hyp['max_detect'])
        return outputs, data_dict

    def process_stats(self, predictions, data_dict):
        if hasattr_not_none(self, 'writer') and self.visual_image:
            images = data_dict['x']
            WRITER.add_batch_images_predictions_detect(self.writer, 'test_pred', data_dict['index'],
                                                       images, predictions, self.epoch)
        # add metrics data to json_dt
        shape_converts = data_dict['shape_converts']
        img_ids = data_dict['img_ids']
        for idx, (p, img_id) in enumerate(zip(predictions, img_ids)):
            p[:, :4] = rescale_xyxy(p[:, :4], shape_converts[idx])  # to original image shape and is real
            p[:, :4] = xyxy2x1y1wh(p[:, :4])
            self.append_json_dt(p[:, :4], p[:, 4], p[:, 5], img_id, data_dict['json_dt'])
        return data_dict

    def postprocess(self, data_dict):
        data_dict = super(ValDetect, self).postprocess(data_dict)
        # when json_dt empty to avoid bug
        self.empty_append(data_dict['json_dt'])

        if self.epoch == -1:  # save json_dt in the test
            with open(self.coco_json['dt'], 'w') as f:
                json.dump(data_dict['json_dt'], f)
            LOGGER.info(f"Save coco_dt json {self.coco_json['dt']} successfully")
        return data_dict


def parse_args_detect(known: bool = False):
    r"""
    Parse args for training.
    Args:
        known: bool = True or False, Default=False
            parser will get two namespace which the second is unknown args, if known=True.

    Return namespace(for setting args)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model',
                        default={'load_key': 'model', 'state_dict_operation': True, 'load': 'state_dict'},
                        help='')
    parser.add_argument('--weights', type=str, default=str(ROOT / 'models/yolov5/yolov5s_v6.pt'), help='')
    parser.add_argument('--half', type=bool, default=False, help='')
    parser.add_argument('--device', type=str, default='0', help='cpu or cuda:0 or 0')
    parser.add_argument('--inc', type=int, default=3, help='')
    parser.add_argument('--image_size', type=int, default=640, help='')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--hyp', type=str, default=str(ROOT / 'data/hyp/hyp_detect_train.yaml'), help='')
    parser.add_argument('--workers', type=int, default=0, help='')
    parser.add_argument('--pin_memory', type=bool, default=False, help='')
    parser.add_argument('--datasets', type=str, default=str(ROOT / 'mine/data/datasets/Customdatasets.yaml'), help='')
    parser.add_argument('--task', type=str, default='val', help='test val train')
    parser.add_argument('--save_name', type=str, default='exp', help='')
    parser.add_argument('--save_path', type=str, default=str(ROOT / 'runs/val/detect'), help='')
    namespace = parser.parse_known_args()[0] if known else parser.parse_args()
    return namespace


@timer
def val_detection():
    arguments = parse_args_detect()
    valer = ValDetect(arguments)
    valer.val()


if __name__ == '__main__':
    val_detection()
