r"""
For training model.
Consist of some Trainers.
"""

import argparse
import torch.nn as nn

from pathlib import Path
from torch.optim import SGD
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import StepLR

from utils.loss import LossDetectYolov5
from models.yolov5.yolov5_v6 import yolov5s_v6
from metaclass.metatrainer import MetaTrainDetect
from utils.datasets import get_and_check_datasets_yaml, DatasetDetect
from utils.general import timer, load_all_yaml, save_all_yaml, init_seed, select_one_device

from val import ValDetect

r"""Set Global Constant for file save and load"""
ROOT = Path.cwd()  # **/visual-framework root directory


class TrainDetect(MetaTrainDetect):
    r"""Trainer for detection, built by mixins"""

    def __init__(self, args):
        super(TrainDetect, self).__init__(args)

        # Set val class
        self.val_class = ValDetect

        # TODO design a way to get all parameters in train setting for research if possible

        # Get save_dict
        self.save_dict, self.writer = self.get_save_path(('hyp', 'hyp.yaml'),
                                                         ('args', 'args.yaml'),
                                                         ('datasets', 'datasets.yaml'),
                                                         ('all_results', 'all_results.txt'),
                                                         ('all_class_results', 'all_class_results.txt'),
                                                         ('last', 'weights/last.pt'),
                                                         ('best', 'weights/best.pt'),
                                                         logfile='logger.log')

        # Set one device
        self.device = select_one_device(self.device)  # requires model, images, labels .to(self.device)
        self.cuda = (self.device != 'cpu')

        # Load hyp yaml
        self.hyp = load_all_yaml(self.hyp)

        # Initialize or auto seed manual and save in self.hyp
        self.hyp['seed'] = init_seed(self.hyp['seed'])

        # Get datasets path dict
        self.datasets = get_and_check_datasets_yaml(self.datasets)

        # Save yaml dict
        save_all_yaml((vars(args), self.save_dict['args']),
                      (self.hyp, self.save_dict['hyp']),
                      (self.datasets, self.save_dict['datasets']))
        args = self.empty()

        # TODO auto compute anchors when anchors is None in self.datasets

        # Load checkpoint
        self.checkpoint = self.load_checkpoint(self.weights)

        # TODO upgrade DP DDP

        # Initialize or load model
        self.model = self.load_model(yolov5s_v6(self.datasets['anchors'], self.inc, self.datasets['nc'],
                                                self.image_size), load=self._load_model)

        # Set parameter groups to for the optimizer
        self.param_groups = self.set_param_groups((('bias', nn.Parameter, {}),
                                                   ('weight', nn.BatchNorm2d, {}),
                                                   ('weight', nn.Parameter, {'weight_decay': self.hyp['weight_decay']})
                                                   ))

        # Initialize and load optimizer
        self.optimizer = self.load_optimizer(SGD(self.param_groups,
                                                 lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True),
                                             load=self._load_optimizer)
        self.param_groups = self.empty()

        # Initialize and load lr_scheduler
        self.lr_scheduler = self.load_lr_scheduler(StepLR(self.optimizer, 30), load=self._load_lr_scheduler)
        # TODO set lr_scheduler args to self.hyp

        # Initialize and load GradScaler
        self.scaler = self.load_gradscaler(GradScaler(enabled=self.cuda), load=self._load_gradscaler)

        # Initialize or load start_epoch
        self.start_epoch = self.load_start_epoch(load=self._load_start_epoch)

        # Initialize or load best_fitness
        self.best_fitness = self.load_best_fitness(load=self._load_best_fitness)

        # Empty self.checkpoint when load finished
        self.checkpoint = self.empty()

        # Get dataloader for training testing
        self.train_dataloader = self.get_dataloader(DatasetDetect, 'train', augment=self.augment,
                                                    data_augment=self.data_augment, shuffle=self.shuffle)
        self.val_dataloader = self.get_dataloader(DatasetDetect, 'val')
        self.test_dataloader = self.get_dataloader(DatasetDetect, 'test')

        # TODO upgrade warmup

        # Get loss function
        self.loss_fn = self.get_loss_fn(LossDetectYolov5)

        # To save results of training and validating
        self.results = self.get_results_dict()


class TrainClassify:
    def __init__(self, args):
        super(TrainClassify, self).__init__()
        pass


def parse_args_detect(known: bool = False):
    r"""
    Parse args for training.
    Args:
        known: bool = True or False, Default=False
            parser will get two namespace which the second is unknown args, if known=True.

    Return namespace(for setting args)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard', type=bool, default=True, help='')
    parser.add_argument('--visual_image', type=bool, default=True,
                        help='whether make images visual in tensorboard')
    parser.add_argument('--visual_graph', type=bool, default=False,
                        help='whether make model graph visual in tensorboard')
    parser.add_argument('--weights', type=str, default=str(ROOT / ''), help='')
    parser.add_argument('--device', type=str, default='0', help='cpu or cuda:0 or 0')
    parser.add_argument('--epochs', type=int, default=300, help='epochs for training')
    parser.add_argument('--batch_size', type=int, default=2, help='')
    parser.add_argument('--workers', type=int, default=0, help='')
    parser.add_argument('--shuffle', type=bool, default=True, help='')
    parser.add_argument('--pin_memory', type=bool, default=True, help='')
    parser.add_argument('--datasets', type=str, default=str(ROOT / 'data/datasets/Mydatasets.yaml'), help='')
    parser.add_argument('--name', type=str, default='exp', help='')
    parser.add_argument('--save_path', type=str, default=str(ROOT / 'runs/train'), help='')
    parser.add_argument('--hyp', type=str, default=str(ROOT / 'data/hyp/hyp_detect_train.yaml'), help='')
    parser.add_argument('--augment', type=bool, default=False, help='whether random augment image')
    parser.add_argument('--data_augment', type=str, default='mosaic',
                        help='the kind of data augmentation mosaic / mixup / cutout')
    parser.add_argument('--inc', type=int, default=3, help='')
    parser.add_argument('--image_size', type=int, default=640, help='')
    parser.add_argument('--load_model', type=str, default=None, help='')
    parser.add_argument('--load_optimizer', type=bool, default=False, help='')
    parser.add_argument('--load_lr_scheduler', type=bool, default=False, help='')
    parser.add_argument('--load_gradscaler', type=bool, default=False, help='')
    parser.add_argument('--load_start_epoch', type=str, default=None, help='')
    parser.add_argument('--load_best_fitness', type=bool, default=False, help='')
    namespace = parser.parse_known_args()[0] if known else parser.parse_args()
    return namespace


@timer
def train_detection():
    arguments = parse_args_detect()
    trainer = TrainDetect(arguments)
    trainer.train()


if __name__ == '__main__':
    train_detection()

    # in the future
    # TODO colour str
    # TODO learn moviepy library sometimes

    # when need because it is complex
    # TODO add FLOPs compute module for model
    # TODO auto compute anchors

    # next work
    # TODO add necessary functions
    # TODO confusion matrix needed
    # TODO add plot curve functions for visual results
    # TODO add pycocotools
