r"""
For training model.
Consist of some Trainers.
"""

import argparse
import torch
import torch.nn as nn

from pathlib import Path
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler
from typing import Optional

# import from mylib below
from models import ModelDetect
from val import ValDetect
from utils import \
    LOGGER, timer, \
    load_all_yaml, save_all_yaml, init_seed, select_one_device, get_and_check_datasets_yaml, \
    DatasetDetect, \
    LossDetectYolov5, \
    SetSavePathMixin, SaveCheckPointMixin, LoadAllCheckPointMixin, DataLoaderMixin, LossMixin, TrainDetectMixin, \
    ResultsDealDetectMixin

r"""Set Global Constant for file save and load"""
ROOT = Path.cwd()  # **/visual-framework root directory


# TODO design a meta class for training of detection
class TrainDetect(
    SetSavePathMixin,  # set and get the paths for saving file of training
    LoadAllCheckPointMixin,  # load the config of the model trained before and others for training
    DataLoaderMixin,  # get dataloader
    LossMixin,
    TrainDetectMixin,  # for training
    ResultsDealDetectMixin,
    SaveCheckPointMixin,
):
    r"""Trainer for detection, built by mixins"""

    def __init__(self, args):
        super(TrainDetect, self).__init__()
        LOGGER.info('Initializing trainer for detection')
        self.hyp = args.hyp
        self.inc = args.inc
        self.name = args.name
        self.epoch = None
        self.device = args.device
        self.epochs = args.epochs
        self.workers = args.workers
        self.shuffle = args.shuffle
        self.weights = Path(args.weights)
        self.save_path = Path(args.save_path)
        self.image_size = args.image_size
        self.batch_size = args.batch_size
        self.pin_memory = args.pin_memory
        self.tensorboard = args.tensorboard
        self.datasets_path = args.datasets_path

        # for load way
        self._load_model = args.load_model
        self._load_optimizer = args.load_optimizer
        self._load_gradscaler = args.load_gradscaler
        self._load_start_epoch = args.load_start_epoch
        self._load_best_fitness = args.load_best_fitness
        self._load_lr_scheduler = args.load_lr_scheduler

        self.writer = None  # TODO it must be existed in MetaTrainer

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
        self.datasets = get_and_check_datasets_yaml(self.datasets_path)

        # Save yaml dict
        save_all_yaml((vars(args), self.save_dict['args']),
                      (self.hyp, self.save_dict['hyp']),
                      (self.datasets, self.save_dict['datasets']))
        args = self._empty_none()

        # TODO auto compute anchors when anchors is None in self.datasets

        # Load checkpoint(has to self.device)
        self.checkpoint = self.load_checkpoint(self.weights)

        # TODO upgrade DP DDP

        # Initialize or load model(has to self.device)
        self.model = self.load_model(ModelDetect(self.inc, self.datasets['nc'], self.datasets['anchors'],
                                                 image_size=self.image_size), load=self._load_model)

        # Set parameter groups to add to the optimizer
        self.param_groups = self.set_param_groups((('bias', nn.Parameter, {}),
                                                   ('weight', nn.BatchNorm2d, {}),
                                                   ('weight', nn.Parameter, {'weight_decay': self.hyp['weight_decay']})
                                                   ))

        # Initialize and load optimizer
        self.optimizer = self.load_optimizer(SGD(self.param_groups,
                                                 lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True),
                                             load=self._load_optimizer)
        self.param_groups = self._empty_none()

        # Initialize and load lr_scheduler
        self.lr_scheduler = self.load_lr_scheduler(StepLR(self.optimizer, 30), load=self._load_lr_scheduler)
        # TODO set lr_scheduler args to self.hyp

        # Initialize and load GradScaler
        self.scaler = self.load_gradscaler(GradScaler(enabled=self.cuda), load=self._load_gradscaler)

        # Initialize or load start_epoch
        self.start_epoch = self.load_start_epoch(load=self._load_start_epoch)

        # Initialize or load best_fitness
        self.best_fitness = self.load_best_fitness(load=self._load_best_fitness)

        # self.checkpoint to None when load finished
        self.checkpoint = self._empty_none()

        # Get dataloader for training testing
        self.train_dataloader = self.get_dataloader(DatasetDetect, 'train', shuffle=self.shuffle)
        self.val_dataloader = self.get_dataloader(DatasetDetect, 'val')
        self.test_dataloader = self.get_dataloader(DatasetDetect, 'test')

        # TODO upgrade warmup

        # Get loss function
        self.loss_fn = self.get_loss_fn(LossDetectYolov5)

        # To save results of training and validating
        self.results = self.get_results_dict(('all_results', []),
                                             ('all_class_results', []))

        LOGGER.info('Initialize trainer successfully')

    def train(self):
        LOGGER.info('Start training')
        for self.epoch in range(self.start_epoch, self.epochs):
            loss_all, loss_name = self.train_one_epoch()
            self._log_results(loss_all, loss_name)
            results_val = self._val_training()
            results, results_for_best = self.deal_results_memory((loss_all, loss_name), results_val)
            self.add_data_results(('all_results', results[0]),
                                  ('all_class_results', results[1]))
            self.save_checkpoint(results_for_best)
            # TODO maybe need a auto stop function for bad training

        self.model = self._empty_none()
        test_results = self._test_trained()
        results, _ = self.deal_results_memory((None, None), test_results)
        self.add_data_results(('all_results', results[0]),
                              ('all_class_results', results[1]))
        self.save_all_results()
        self.writer.flush()
        self.writer.close()
        torch.cuda.empty_cache()
        LOGGER.info('Finished training')

    @torch.no_grad()
    def _val_training(self):
        valer = ValDetect(last=False, model=self.model, half=True, dataloader=self.val_dataloader,
                          loss_fn=self.loss_fn, cls_names=self.datasets['names'], epoch=self.epoch, writer=self.writer)
        results = valer.val()
        return results

    @torch.inference_mode()
    def _test_trained(self):
        self.epoch = -1
        self.checkpoint = self._empty_none()
        self.checkpoint = self.load_checkpoint(self.save_dict['best'])
        if self.checkpoint is None:
            self.checkpoint = self.load_checkpoint(self.save_dict['last'])
            LOGGER.info('Load last.pt for validating because of no best.pt')
        else:
            LOGGER.info('Load best.pt for validating')
        self.model = self.load_model(load='model')
        if self.test_dataloader is not None:
            dataloader = self.test_dataloader
        else:
            dataloader = self.val_dataloader
        valer = ValDetect(last=True, model=self.model, half=False, dataloader=dataloader,
                          loss_fn=self.loss_fn, cls_names=self.datasets['names'], epoch=self.epoch, writer=self.writer)
        results = valer.val()
        return results

    def _log_results(self, loss_all, loss_name):
        LOGGER.debug(f'Training epoch{self.epoch}: {loss_name} is {loss_all}')

    @staticmethod
    def _empty_none(empty=None):
        return empty


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
    _str_or_None = Optional[str]
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard', type=bool, default=True, help='')
    parser.add_argument('--weights', type=str, default=str(ROOT / ''), help='')
    parser.add_argument('--device', type=str, default='0', help='cpu or cuda:0 or 0')
    parser.add_argument('--epochs', type=int, default=10, help='epochs for training')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--workers', type=int, default=0, help='')
    parser.add_argument('--shuffle', type=bool, default=True, help='')
    parser.add_argument('--pin_memory', type=bool, default=True, help='')
    parser.add_argument('--datasets_path', type=str, default=str(ROOT / 'data/datasets/Mydatasets.yaml'), help='')
    parser.add_argument('--name', type=str, default='exp', help='')
    parser.add_argument('--save_path', type=str, default=str(ROOT / 'runs/train'), help='')
    parser.add_argument('--hyp', type=str, default=str(ROOT / 'data/hyp/hyp_detect_train.yaml'), help='')
    parser.add_argument('--inc', type=int, default=3, help='')
    parser.add_argument('--image_size', type=int, default=640, help='')
    parser.add_argument('--load_model', type=_str_or_None, default=None, help='')
    parser.add_argument('--load_optimizer', type=bool, default=False, help='')
    parser.add_argument('--load_lr_scheduler', type=bool, default=False, help='')
    parser.add_argument('--load_gradscaler', type=bool, default=False, help='')
    parser.add_argument('--load_start_epoch', type=_str_or_None, default=None, help='')
    parser.add_argument('--load_best_fitness', type=bool, default=False, help='')
    namespace = parser.parse_known_args()[0] if known else parser.parse_args()
    return namespace


@timer
def train_detection():
    arguments = parse_args()
    trainer = TrainDetect(arguments)
    trainer.train()


if __name__ == '__main__':
    train_detection()

    # in the future
    # TODO colour str
    # TODO datasets augment
    # TODO learn moviepy sometimes

    # when need because it is complex
    # TODO auto compute anchors

    # TODO next work
    # TODO Meta Trainer Module
