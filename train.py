r"""
For training model.
Consist of some Trainers.
"""

import argparse
import torch.nn as nn

from pathlib import Path

from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler

# import from mylib below
from models import ModelDetect
from utils import \
    LOGGER, timer, \
    load_all_yaml, save_all_yaml, init_seed, select_one_device, get_and_check_datasets_yaml, \
    DatasetDetect, \
    LossDetectYolov5, \
    SetSavePathMixin, LoadAllCheckPointMixin, DataLoaderMixin, LossMixin, TrainDetectMixin

r"""Set Global Constant for file save and load"""
ROOT = Path.cwd()  # **/visual-framework root directory


class TrainDetect(
    SetSavePathMixin,  # set and get the paths for saving file of training
    LoadAllCheckPointMixin,  # load the config of the model trained before and others for training
    DataLoaderMixin,  # get dataloader
    LossMixin,
    TrainDetectMixin,  # for training
):
    r"""Trainer for detection, built by mixins"""

    def __init__(self, args):
        super(TrainDetect, self).__init__()
        # Get args
        LOGGER.info('Initializing trainer for detection')
        self.device = args.device
        self.save_path = Path(args.save_path)
        self.weights = Path(args.weights)
        self.name = args.name
        self.datasets_path = args.datasets_path
        self.hyp = args.hyp
        self.seed = args.seed
        self.inc = args.inc
        self.image_size = args.image_size
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.shuffle = args.shuffle
        self.pin_memory = args.pin_memory
        # TODO set hyp['load'] for model, optimizer, lr_scheduler etc. in the future
        # TODO design a way to get all parameters in train setting for research

        # Set one device
        self.device = select_one_device(self.device)  # requires model, images, labels .to(self.device)
        self.cuda = (self.device != 'cpu')

        # Initialize or auto seed manual
        init_seed(self.seed)

        # Get save_dict
        self.save_dict = self.get_save_path(('hyp', 'hyp.yaml'),
                                            ('args', 'args.yaml'),
                                            ('datasets', 'datasets.yaml'),
                                            ('results', 'results.txt'),
                                            ('last', 'weights/last.pt'),
                                            ('best', 'weights/best.pt'),
                                            logfile='logger.log')
        # Load hyp yaml
        self.hyp = load_all_yaml(self.hyp)

        # Get datasets path dict
        self.datasets = get_and_check_datasets_yaml(self.datasets_path)

        # Save yaml dict
        save_all_yaml((vars(args), self.save_dict['args']),
                      (self.hyp, self.save_dict['hyp']),
                      (self.datasets, self.save_dict['datasets']))
        del args

        # Load checkpoint(has to self.device)
        self.checkpoint = self.load_checkpoint()

        # TODO upgrade DP DDP
        # TODO check whether start epoch set

        # Initialize or load model(has to self.device)
        self.model = self.load_model(ModelDetect(self.inc, self.datasets['nc']), load=None)

        # Set parameter groups to add to the optimizer
        self.param_groups = self.set_param_groups((('bias', nn.Parameter, {}),
                                                   ('weight', nn.BatchNorm2d, {}),
                                                   ('weight', nn.Parameter, {'weight_decay': self.hyp['weight_decay']})
                                                   ))

        # Initialize and load optimizer
        self.optimizer = self.load_optimizer(SGD(self.param_groups.pop(1)['params'],
                                                 lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True),
                                             load=False)

        # Initialize and load lr_scheduler
        self.lr_scheduler = self.load_lr_scheduler(StepLR(self.optimizer, 30), load=False)

        # Initialize and load GradScaler
        self.scaler = self.load_gradscaler(GradScaler(enabled=self.cuda), load=False)

        # Initialize or load start_epoch
        self.start_epoch = self.load_start_epoch(load=None)

        # Initialize or load best_fitness
        self.best_fitness = self.load_best_fitness(load=False)

        # Delete self.checkpoint when load finished
        del self.checkpoint

        # Get dataloader for training testing
        self.train_dataloader = self.get_dataloader(DatasetDetect, 'train')
        # self.val_dataloader = self.get_dataloader(DatasetDetect, 'val')
        # self.test_dataloader = self.get_dataloader(DatasetDetect, 'test')

        # TODO upgrade warmup

        # Get loss function
        self.loss_fn = self.get_loss_fn(LossDetectYolov5)
        LOGGER.info('Initialize trainer successfully')

    def train(self):
        LOGGER.info('Start training')
        for self.epoch in range(self.start_epoch, self.epochs):
            loss_all, loss_name = self.train_one_epoch()
            LOGGER.debug(f'Training: {loss_name} is {loss_all}')


class TrainClassify(SetSavePathMixin):
    def __init__(self, args):
        super(TrainClassify, self).__init__()
        pass


def parse_args(known: bool = False):
    r"""
    Parse args for training.
    Args:
        known: bool = True or False, Default=False
            parser will get two namespace which the second is unknown args, if known=True.

    Return namespace(for setting args)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=str(ROOT / ''), help='')
    parser.add_argument('--device', type=str, default='0', help='cpu or cuda:0 or 0')
    parser.add_argument('--epochs', type=int, default=100, help='epochs for training')
    parser.add_argument('--batch_size', type=int, default=2, help='')
    parser.add_argument('--workers', type=int, default=0, help='')
    parser.add_argument('--shuffle', type=bool, default=True, help='')
    parser.add_argument('--pin_memory', type=bool, default=True, help='')
    parser.add_argument('--datasets_path', type=str, default=str(ROOT / 'data/datasets/Mydatasets.yaml'), help='')
    parser.add_argument('--name', type=str, default='exp', help='')
    parser.add_argument('--save_path', type=str, default=str(ROOT / 'runs/train'), help='')
    parser.add_argument('--hyp', type=str, default=str(ROOT / 'data/hyp/hyp_detect_train.yaml'), help='')
    parser.add_argument('--seed', type=int, default=None, help='None is auto')
    parser.add_argument('--inc', type=int, default=3, help='')
    parser.add_argument('--image_size', type=int, default=640, help='')
    namespace = parser.parse_known_args()[0] if known else parser.parse_args()
    return namespace


@timer
def train_detection():
    arguments = parse_args()
    trainer = TrainDetect(arguments)
    trainer.train()


if __name__ == '__main__':
    train_detection()
