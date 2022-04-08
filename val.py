r"""
For validating model.
Consist of some Valers.
"""

import argparse

from pathlib import Path

from models.yolov5.yolov5_v6 import yolov5s_v6
from utils.log import add_log_file
from utils.loss import LossDetectYolov5
from utils.datasets import get_and_check_datasets_yaml, DatasetDetect
from utils.general import timer, select_one_device, save_all_yaml, load_all_yaml
from metaclass.metavaler import MetaValDetect

__all__ = ['ValDetect']

r"""Set Global Constant for file save and load"""
ROOT = Path.cwd()  # **/visual-framework root directory


# TODO whether the different batch size comparing to training will affect the val mAP

class ValDetect(MetaValDetect):
    def __init__(self, args=None,
                 last=True, model=None, writer=None,
                 half=True, dataloader=None, loss_fn=None, cls_names=None, epoch=None, visual_image=None,
                 coco_eval=None):
        super(ValDetect, self).__init__(last, model, writer, half, dataloader,
                                        loss_fn, cls_names, epoch, visual_image, coco_eval, args)

    def set_self_parameters_val(self, args):
        super(ValDetect, self).set_self_parameters_val(args)
        self.path_dict = self.get_save_path(('hyp', 'hyp.yaml'),
                                            ('logger', 'logger.log'),
                                            ('args', 'args.yaml'),
                                            ('datasets', 'datasets.yaml'),
                                            ('json_gt', 'json_gt.json'),
                                            ('json_dt', 'json_dt.json'),
                                            ('coco_results', 'coco_results.json'))
        # Add FileHandler for logger
        add_log_file(self.path_dict['logger'])

        self.coco_eval = (self.path_dict['json_gt'], self.path_dict['json_dt'])
        self.device = select_one_device(self.device)
        self.hyp = load_all_yaml(self.hyp)
        self.datasets = get_and_check_datasets_yaml(self.datasets)
        self.cls_names = self.datasets['names']
        save_all_yaml((vars(args), self.path_dict['args']),
                      (self.hyp, self.path_dict['hyp']),
                      (self.datasets, self.path_dict['datasets']))
        self.checkpoint = self.load_checkpoint(self.weights)

        # TODO load='state_dict'
        self.model = self.load_model(yolov5s_v6(self.inc, self.datasets['nc'], self.datasets['anchors'],
                                                self.image_size))

        self.loss_fn = LossDetectYolov5(self.model, self.hyp)
        self.dataloader = self.set_dataloader(DatasetDetect(self.datasets, self.task, self.image_size,
                                                            json_gt=self.path_dict['json_gt']))


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
    parser.add_argument('--name', type=str, default='exp', help='')
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
