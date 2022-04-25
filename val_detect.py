r"""
For validating model.
Consist of some Valers.
"""

import torch
import argparse

from pathlib import Path

from utils import WRITER
from utils.log import LOGGER, add_log_file, log_results
from utils.loss import LossDetectYolov5
from utils.datasets import get_and_check_datasets_yaml, DatasetDetect
from utils.general import timer, select_one_device, save_all_yaml, load_all_yaml
from metaclass.metavaler import MetaValDetect
from models.yolov5.yolov5_v6 import yolov5s_v6

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


class ValDetect(_Args, MetaValDetect):
    def __init__(self, args=None, model=None, writer=None, half=True, dataloader=None, loss_fn=None,
                 cls_names=None, epoch=None, visual_image=None, coco_json=None, hyp=None):
        # all need
        self.seen = 0
        self.time = 0.0
        self.writer = writer
        self.training = model is not None

        if self.training:
            self.hyp = hyp
            self.half = half
            self.epoch = epoch
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
            self.epoch = -1  # TODO a bug about super in subclass
            self.visual_image = None  # TODO a bug about super in subclass

            self.path_dict = self.get_save_path(('hyp', 'hyp.yaml'),
                                                ('logger', 'logger.log'),
                                                ('args', 'args.yaml'),
                                                ('datasets', 'datasets.yaml'),
                                                ('json_gt', 'json_gt.json'),
                                                ('json_dt', 'json_dt.json'),
                                                ('coco_results', 'coco_results.json'))
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
                load='state_dict'
            )

            self.loss_fn = LossDetectYolov5(self.model, self.hyp)
            self.dataloader = self.set_dataloader(
                DatasetDetect(self.datasets, self.task, self.image_size, coco_gt=self.coco_json['test']))

    @torch.no_grad()
    def val_training(self):
        self.model.eval()
        json_dt = self.val_once(('total_loss', 'bbox_loss', 'class_loss', 'object_loss'))
        img_ids = self.dataloader.dataset.indices

        if self.epoch != -1:
            coco_eval, coco_stats = self.coco_evaluate(self.coco_json['val'], json_dt, img_ids, 'bbox')
            self._log_writer(coco_stats, writer=True)
        else:
            coco_eval, coco_stats = self.coco_evaluate(self.coco_json['test'], json_dt, img_ids, 'bbox',
                                                       print_result=True)
            self._log_writer(coco_stats, writer=True)
        # TODO confusion matrix needed
        self.model.float()
        return coco_eval, coco_stats

    @torch.inference_mode()
    def val(self):
        self.model.eval()
        json_dt = self.val_once(('total_loss', 'bbox_loss', 'class_loss', 'object_loss'))
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


def parse_args_detect(known: bool = False):
    r"""
    Parse args for training.
    Args:
        known: bool = True or False, Default=False
            parser will get two namespace which the second is unknown args, if known=True.

    Return namespace(for setting args)
    """
    parser = argparse.ArgumentParser()
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
