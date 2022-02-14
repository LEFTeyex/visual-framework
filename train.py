r"""
For training model.
Consist of some Trainers.
"""

import argparse
import torch.nn as nn

from pathlib import Path
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

# import from mylib below
from models import ModelDetect
from utils import \
    load_all_yaml, save_all_yaml, init_seed, select_one_device, \
    LOGGER, \
    SetSavePathMixin, LoadAllCheckPointMixin

r"""Set Global Constant for file save and load"""
ROOT = Path.cwd()  # **/visual-framework root directory


class TrainDetect(
    SetSavePathMixin,  # set and get the paths for saving file of training
    LoadAllCheckPointMixin  # load the config of the model trained before and others for training
):
    r"""Trainer for detection, built by mixins"""

    def __init__(self, args):
        super(TrainDetect, self).__init__()
        # Get args
        self.device = args.device
        self.save_path = Path(args.save_path)
        self.weights = Path(args.weights)
        self.name = args.name
        self.hyp = args.hyp
        self.seed = args.seed
        self.inc = args.inc
        self.num_class = args.num_class
        self.image_size = args.image_size
        self.epochs = args.epochs
        # TODO set hyp['load'] for model, optimizer, lr_scheduler etc. in the future
        # TODO design a way to get all parameters in train setting for research

        # Set one device
        self.device = select_one_device(self.device)

        # Initialize or auto seed manual
        init_seed(self.seed)

        # Get save_dict
        self.save_dict = self.get_save_path(('hyp', 'hyp.yaml'),
                                            ('args', 'args.yaml'),
                                            ('results', 'results.txt'),
                                            ('last', 'weights/last.pt'),
                                            ('best', 'weights/best.pt'),
                                            logfile='logger.log')
        # Load hyp yaml
        self.hyp = load_all_yaml(self.hyp)

        # Save yaml dict
        save_all_yaml((vars(args), self.save_dict['args']),
                      (self.hyp, self.save_dict['hyp']))

        # Load checkpoint(has to self.device)
        self.checkpoint = self.load_checkpoint()

        # Initialize or load model(has to self.device)
        self.model = self.load_model(ModelDetect(self.inc, self.num_class), load=None)

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
        self.scheduler = self.load_lr_scheduler(StepLR(self.optimizer, 30), load=False)

        # Initialize or load start_epoch
        self.start_epoch = self.load_start_epoch(load=None)

        # Initialize or load best_fitness
        self.best_fitness = self.load_best_fitness(load=False)

        # load finished and delete self.checkpoint
        del self.checkpoint

        LOGGER.info('Initialize trainer successfully')


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
    parser.add_argument('--epochs', type=int, default='100', help='epochs for training')
    parser.add_argument('--name', type=str, default='exp', help='')
    parser.add_argument('--save_path', type=str, default=str(ROOT / 'runs/train'), help='')
    parser.add_argument('--hyp', type=str, default=str(ROOT / 'data/hyp/hyp_detect_train.yaml'), help='')
    parser.add_argument('--seed', type=int, default=None, help='None is auto')
    parser.add_argument('--inc', type=int, default=3, help='')
    parser.add_argument('--num_class', type=int, default=1, help='')
    parser.add_argument('--image_size', type=int, default=640, help='')
    namespace = parser.parse_known_args()[0] if known else parser.parse_args()
    return namespace


def main():
    arguments = parse_args()
    train = TrainDetect(arguments)


if __name__ == '__main__':
    main()
