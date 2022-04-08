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
from torchvision.transforms import transforms
from torchvision.datasets import MNIST

from utils.log import add_log_file
from utils.loss import LossDetectYolov5
from models.yolov5.yolov5_v6 import yolov5s_v6
from metaclass.metatrainer import MetaTrainDetect, MetaTrainClassify
from utils.datasets import get_and_check_datasets_yaml, DatasetDetect
from utils.general import timer, load_all_yaml, save_all_yaml, init_seed, select_one_device

from val import ValDetect

from mine.SmartNet.models.smartnet import SmartNet

r"""Set Global Constant for file save and load"""
ROOT = Path.cwd()  # **/visual-framework root directory


class TrainDetect(MetaTrainDetect):
    r"""Trainer for detection, built by mixins"""

    def __init__(self, args):
        # init self.* by args
        super(TrainDetect, self).__init__(args)

        # TODO design a way to get all parameters in train setting for research if possible

        # Get path_dict
        self.path_dict = self.get_save_path(('hyp', 'hyp.yaml'),
                                            ('logger', 'logger.log'),
                                            ('writer', 'tensorboard'),
                                            ('args', 'args.yaml'),
                                            ('datasets', 'datasets.yaml'),
                                            ('all_results', 'all_results.txt'),
                                            ('all_class_results', 'all_class_results.txt'),
                                            ('last', 'weights/last.pt'),
                                            ('best', 'weights/best.pt'),
                                            ('json_gt', 'json_gt.json'),
                                            ('json_dt', 'json_dt.json'),
                                            ('coco_results', 'coco_results.json'))

        # Add FileHandler for logger
        add_log_file(self.path_dict['logger'])

        # Set tensorboard writer
        self.writer = self.set_tensorboard_writer(self.path_dict['writer'])

        # Set coco eval config
        self.coco_eval = (self.path_dict['json_gt'], self.path_dict['json_dt'])

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
        save_all_yaml((vars(args), self.path_dict['args']),
                      (self.hyp, self.path_dict['hyp']),
                      (self.datasets, self.path_dict['datasets']))
        args = self.empty()

        # TODO auto compute anchors when anchors is None in self.datasets

        # Load checkpoint
        self.checkpoint = self.load_checkpoint(self.weights)

        # TODO upgrade DP DDP

        # Initialize or load model
        self.model = self.load_model(yolov5s_v6(self.inc, self.datasets['nc'], self.datasets['anchors'],
                                                self.image_size), load=self._load_model)

        # Unfreeze model
        self.unfreeze_model()

        # Freeze layers of model
        self.freeze_layers(self.freeze_names)

        # Set parameter groups list to for the optimizer
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
        self.lr_scheduler = self.load_lr_scheduler(StepLR(self.optimizer, 50), load=self._load_lr_scheduler)
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
        self.train_dataloader = self.set_dataloader(
            DatasetDetect(self.datasets, 'train', self.image_size, self.augment, self.data_augment, self.hyp),
            shuffle=self.shuffle)

        if self.datasets['test'] is not None:
            self.val_dataloader = self.set_dataloader(
                DatasetDetect(self.datasets, 'val', self.image_size, self.augment, self.data_augment, self.hyp))
            self.test_dataloader = self.set_dataloader(
                DatasetDetect(self.datasets, 'test', self.image_size, self.augment, self.data_augment, self.hyp,
                              json_gt=self.path_dict['json_gt']))
        else:
            self.val_dataloader = self.set_dataloader(
                DatasetDetect(self.datasets, 'val', self.image_size, self.augment, self.data_augment, self.hyp))
            self.test_dataloader = None

        # TODO upgrade warmup

        # Get loss function
        self.loss_fn = LossDetectYolov5(self.model, self.hyp)

        # To save results of training and validating
        self.results = self.get_results_dict()

        # Set val class
        self.val_class = ValDetect


class TrainClassify(MetaTrainClassify):
    def __init__(self, args):
        super(TrainClassify, self).__init__(args)

        # Get path_dict
        self.path_dict = self.get_save_path(('hyp', 'hyp.yaml'),
                                            ('logger', 'logger.log'),
                                            ('writer', 'tensorboard'),
                                            ('args', 'args.yaml'),
                                            ('datasets', 'datasets.yaml'),
                                            ('all_results', 'all_results.txt'),
                                            ('last', 'weights/last.pt'),
                                            ('best', 'weights/best.pt'))

        # Add FileHandler for logger
        add_log_file(self.path_dict['logger'])

        # Set tensorboard writer
        self.writer = self.set_tensorboard_writer(self.path_dict['writer'])

        # Set one device
        self.device = select_one_device(self.device)  # requires model, images, labels .to(self.device)
        self.cuda = (self.device != 'cpu')

        # Load hyp yaml
        self.hyp = load_all_yaml(self.hyp)

        # Initialize or auto seed manual and save in self.hyp
        self.hyp['seed'] = init_seed(self.hyp['seed'])

        # Get datasets path dict
        self.datasets = load_all_yaml(self.datasets)

        # Save yaml dict
        save_all_yaml((vars(args), self.path_dict['args']),
                      (self.hyp, self.path_dict['hyp']),
                      (self.datasets, self.path_dict['datasets']))
        args = self.empty()

        # TODO auto compute anchors when anchors is None in self.datasets

        # Load checkpoint
        self.checkpoint = self.load_checkpoint(self.weights)

        # TODO upgrade DP DDP

        # Initialize or load model
        self.model = self.load_model(SmartNet(self.inc, self.datasets['nc'], self.image_size, self.channels,
                                              invalid=0.01, num_add=5, add_cut_percentage=0.9,
                                              act='relu', device=self.device))

        # Unfreeze model
        self.unfreeze_model()

        # Freeze layers of model
        self.freeze_layers(self.freeze_names)

        # Set parameter groups list to for the optimizer
        self.param_groups = self.set_param_groups((('weight', nn.Parameter, {'weight_decay': self.hyp['weight_decay']}),
                                                   ))

        # Initialize and load optimizer
        self.optimizer = self.load_optimizer(SGD(self.param_groups,
                                                 lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True),
                                             load=self._load_optimizer)
        self.param_groups = self.empty()

        # Initialize and load lr_scheduler
        self.lr_scheduler = self.load_lr_scheduler(StepLR(self.optimizer, 20), load=self._load_lr_scheduler)
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
        transform = transforms.Compose([transforms.ToTensor()])

        self.train_dataloader = self.set_dataloader(MNIST(self.datasets['path'], self.datasets['train'], transform),
                                                    shuffle=self.shuffle)

        if self.datasets['test'] is not None:
            self.val_dataloader = self.set_dataloader(MNIST(self.datasets['path'], self.datasets['val'], transform))
            self.test_dataloader = self.set_dataloader(MNIST(self.datasets['path'], self.datasets['test'], transform))
        else:
            self.val_dataloader = self.set_dataloader(MNIST(self.datasets['path'], self.datasets['val'], transform))
            self.test_dataloader = None

        # TODO upgrade warmup

        # Get loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # To save results of training and validating
        self.results =

        # Set val class
        self.val_class =


def parse_args_detect(known: bool = False):
    r"""
    Parse args for training.
    Args:
        known: bool = True or False, Default=False
            parser will get two namespace which the second is unknown args, if known=True.

    Returns:
        namespace(for setting args)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard', type=bool, default=True, help='')
    parser.add_argument('--visual_image', type=bool, default=True,
                        help='whether make images visual in tensorboard')
    parser.add_argument('--visual_graph', type=bool, default=False,
                        help='whether make model graph visual in tensorboard')
    parser.add_argument('--weights', type=str, default=str(ROOT / 'models/yolov5/yolov5s_v6.pt'), help='')
    parser.add_argument('--freeze_names', type=list, default=['backbone', 'neck'],
                        help='name of freezing layers in model')
    parser.add_argument('--device', type=str, default='0', help='cpu or cuda:0 or 0')
    parser.add_argument('--epochs', type=int, default=1, help='epochs for training')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--workers', type=int, default=0, help='')
    parser.add_argument('--shuffle', type=bool, default=True, help='')
    parser.add_argument('--pin_memory', type=bool, default=False, help='')
    parser.add_argument('--datasets', type=str, default=str(ROOT / 'mine/data/datasets/VOC.yaml'), help='')
    parser.add_argument('--name', type=str, default='exp', help='')
    parser.add_argument('--save_path', type=str, default=str(ROOT / 'runs/train/detect'), help='')
    parser.add_argument('--hyp', type=str, default=str(ROOT / 'data/hyp/hyp_detect_train.yaml'), help='')
    parser.add_argument('--augment', type=bool, default=False, help='whether random augment image')
    parser.add_argument('--data_augment', type=str, default='mosaic',
                        help='the kind of data augmentation mosaic / mixup / cutout')
    parser.add_argument('--inc', type=int, default=3, help='')
    parser.add_argument('--image_size', type=int, default=640, help='')
    parser.add_argument('--channels', type=list, default=[1, 2, 3], help='')
    parser.add_argument('--load_model', type=str, default='state_dict', help='')
    parser.add_argument('--load_optimizer', type=bool, default=False, help='')
    parser.add_argument('--load_lr_scheduler', type=bool, default=False, help='')
    parser.add_argument('--load_gradscaler', type=bool, default=False, help='')
    parser.add_argument('--load_start_epoch', type=str, default=None, help='')
    parser.add_argument('--load_best_fitness', type=bool, default=False, help='')
    namespace = parser.parse_known_args()[0] if known else parser.parse_args()
    return namespace


def parse_args_class(known: bool = False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard', type=bool, default=True, help='')
    parser.add_argument('--visual_image', type=bool, default=True,
                        help='whether make images visual in tensorboard')
    parser.add_argument('--visual_graph', type=bool, default=False,
                        help='whether make model graph visual in tensorboard')
    parser.add_argument('--weights', type=str, default=str(ROOT / 'models/yolov5/yolov5s_v6.pt'), help='')
    parser.add_argument('--freeze_names', type=list, default=['backbone', 'neck'],
                        help='name of freezing layers in model')
    parser.add_argument('--device', type=str, default='0', help='cpu or cuda:0 or 0')
    parser.add_argument('--epochs', type=int, default=1, help='epochs for training')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--workers', type=int, default=0, help='')
    parser.add_argument('--shuffle', type=bool, default=True, help='')
    parser.add_argument('--pin_memory', type=bool, default=False, help='')
    parser.add_argument('--datasets', type=str, default=str(ROOT / 'mine/data/datasets/VOC.yaml'), help='')
    parser.add_argument('--name', type=str, default='exp', help='')
    parser.add_argument('--save_path', type=str, default=str(ROOT / 'runs/train/detect'), help='')
    parser.add_argument('--hyp', type=str, default=str(ROOT / 'data/hyp/hyp_detect_train.yaml'), help='')
    parser.add_argument('--augment', type=bool, default=False, help='whether random augment image')

    parser.add_argument('--inc', type=int, default=3, help='')
    parser.add_argument('--image_size', type=int, default=640, help='')
    parser.add_argument('--load_model', type=str, default='state_dict', help='')
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


@timer
def train_classify():
    arguments = parse_args_class()
    # trainer = TrainClassify(arguments)
    # trainer.train()
    pass


if __name__ == '__main__':
    # train_detection()
    train_classify()

    # in the future
    # TODO colour str
    # TODO learn moviepy library sometimes

    # when need because it is complex
    # TODO add FLOPs compute module for model
    # TODO auto compute anchors

    # next work
    # TODO add threshold for nms and filter predictions
    # TODO add necessary functions
    # TODO confusion matrix needed
    # TODO add plot curve functions for visual results
    # TODO design model structure
