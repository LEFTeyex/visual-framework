r"""
For training model.
Consist of some Trainers.
"""

import torch
import argparse
import torch.nn as nn

from pathlib import Path
from torch.optim import SGD
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import StepLR, LambdaLR

from utils.log import add_log_file, logging_start_finish, logging_initialize, LOGGER
from utils.loss import LossDetectYolov5
from utils.metrics import compute_fitness
from models.yolov5.yolov5_v6 import yolov5s_v6
from metaclass.metatrainer import MetaTrainDetect
from utils.lr_schedulers import WarmUpWithScheduler
from utils.datasets import get_and_check_datasets_yaml, DatasetDetect
from utils.general import timer, load_all_yaml, save_all_yaml, init_seed, select_one_device

from val_detect import ValDetect

r"""Set Global Constant for file save and load"""
ROOT = Path.cwd()  # **/visual-framework root directory


class _Args(object):
    def __init__(self, args):
        self.hyp = args.hyp
        self.inc = args.inc
        self.device = args.device
        self.epochs = args.epochs
        self.weights = args.weights
        self.augment = args.augment
        self.workers = args.workers
        self.shuffle = args.shuffle
        self.datasets = args.datasets
        self.save_name = args.save_name
        self.save_path = args.save_path
        self.image_size = args.image_size
        self.batch_size = args.batch_size
        self.pin_memory = args.pin_memory
        self.tensorboard = args.tensorboard
        self.freeze_names = args.freeze_names
        self.visual_image = args.visual_image
        self.visual_graph = args.visual_graph
        self.data_augment = args.data_augment

        # Set load way
        self._load_model = args.load_model
        self._load_optimizer = args.load_optimizer
        self._load_gradscaler = args.load_gradscaler
        self._load_start_epoch = args.load_start_epoch
        self._load_best_fitness = args.load_best_fitness
        self._load_lr_scheduler = args.load_lr_scheduler


class TrainDetect(_Args, MetaTrainDetect):
    r"""Trainer for detection, built by mixins"""

    @logging_initialize('trainer')
    def __init__(self, args):
        super(TrainDetect, self).__init__(args)
        # TODO design a way to get all parameters in train setting for research if possible

        # Get path_dict
        self.path_dict = self.get_save_path(('hyp', 'hyp.yaml'),
                                            ('logger', 'logger.log'),
                                            ('writer', 'tensorboard'),
                                            ('args', 'args.yaml'),
                                            ('datasets', 'datasets.yaml'),
                                            ('last', 'weights/last.pt'),
                                            ('best', 'weights/best.pt'),
                                            ('json_gt_val', 'json_gt_val.json'),
                                            ('json_gt_test', 'json_gt_test.json'),
                                            ('json_dt', 'json_dt.json'),
                                            ('coco_results', 'coco_results.json'))

        # Add FileHandler for logger
        add_log_file(self.path_dict['logger'])

        # Set tensorboard writer
        self.writer = self.set_tensorboard_writer(self.path_dict['writer'])

        # Set one device
        self.cuda = (self.device != 'cpu')
        self.device = select_one_device(self.device)  # requires (model, images, labels).to(self.device)

        # Get datasets path dict
        self.datasets = get_and_check_datasets_yaml(self.datasets)

        # Load hyp yaml
        self.hyp = load_all_yaml(self.hyp)

        # Initialize or auto seed manual and save in self.hyp
        self.hyp['seed'] = init_seed(self.hyp['seed'])

        nl = int(len(self.datasets['anchors']))
        self.hyp['bbox'] *= 3 / nl
        self.hyp['cls'] *= self.datasets['nc'] / 80 * 3. / nl
        self.hyp['obj'] *= (self.image_size / 640) ** 2 * 3. / nl

        # Save yaml dict
        save_all_yaml(
            (self.hyp, self.path_dict['hyp']),
            (vars(args), self.path_dict['args']),
            (self.datasets, self.path_dict['datasets'])
        )
        args = self.release()

        # TODO auto compute anchors when anchors is None in self.datasets

        # Load checkpoint
        self.checkpoint = self.load_checkpoint(self.weights)

        # TODO upgrade DP DDP

        # Initialize or load model
        self.model = self.load_model(
            yolov5s_v6(self.inc, self.datasets['nc'], self.datasets['anchors'], self.image_size),
            load=self._load_model
        )

        # Unfreeze model
        self.unfreeze_model()

        # Freeze layers of model
        self.freeze_layers(self.freeze_names)

        # Set parameter groups list to for the optimizer
        param_groups = self.set_param_groups(
            (('bias', nn.Parameter, {}),
             ('weight', nn.BatchNorm2d, {}),
             ('weight', nn.Parameter, {'weight_decay': self.hyp['weight_decay']}))
        )

        # Initialize and load optimizer
        self.optimizer = self.load_optimizer(
            SGD(param_groups, lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True),
            load=self._load_optimizer
        )

        param_groups = self.release()

        # Initialize and load lr_scheduler
        if self.hyp['lr_scheduler'] == 'lr_lambda':
            self.lr_scheduler = self.load_lr_scheduler(
                LambdaLR(self.optimizer, lambda x: (1 - x / self.epochs) * (1.0 - self.hyp['lrf']) + self.hyp['lrf']),
                load=self._load_lr_scheduler)
        else:
            self.lr_scheduler = self.load_lr_scheduler(
                StepLR(self.optimizer, self.hyp['step'], self.hyp['gamma']),
                load=self._load_lr_scheduler)

        # Initialize and load GradScaler
        self.scaler = self.load_gradscaler(GradScaler(enabled=self.cuda), load=self._load_gradscaler)

        # Initialize or load start_epoch
        self.start_epoch = self.load_start_epoch(load=self._load_start_epoch)

        # Initialize or load best_fitness
        self.best_fitness = self.load_best_fitness(load=self._load_best_fitness)

        # Release self.checkpoint when load finished
        self.checkpoint = self.release()

        # Get loss function
        self.loss_fn = LossDetectYolov5(self.model, self.hyp)

        # Set coco json path dict
        self.coco_json = {
            'dt': self.path_dict['json_dt'],
            'val': self.path_dict['json_gt_val'],
            'test': self.path_dict['json_gt_test'] if self.datasets['test'] else self.path_dict['json_gt_val']
        }

        # Get dataloader for training validating testing
        self.train_dataloader = self.set_dataloader(
            DatasetDetect(self.datasets, 'train', self.image_size, self.augment, self.data_augment, self.hyp),
            shuffle=self.shuffle)

        self.val_dataloader = self.set_dataloader(
            DatasetDetect(self.datasets, 'val', self.image_size, coco_gt=self.coco_json['val']))

        self.test_dataloader = self.set_dataloader(
            DatasetDetect(self.datasets, 'test', self.image_size, coco_gt=self.coco_json['test'])
        ) if self.datasets['test'] else self.val_dataloader

        self.warmup_lr_scheduler = WarmUpWithScheduler(self.optimizer, self.lr_scheduler,
                                                       len_loader=len(self.train_dataloader),
                                                       warmup_steps=self.hyp['warmup_steps'],
                                                       warmup_start_lr=self.hyp['warmup_start_lr'],
                                                       warmup_mode=self.hyp['warmup_mode'])

        self.val_class = ValDetect

    # TODO upgrade warmup

    @logging_start_finish('Training')
    def train(self):
        for self.epoch in range(self.start_epoch, self.epochs):
            self.train_one_epoch(('total_loss', 'bbox_loss', 'class_loss', 'object_loss'))
            _, coco_stats = self.val_training()

            fitness = compute_fitness(coco_stats[:3], self.hyp['fit_weights'])  # compute fitness for best save
            self.save_checkpoint_best_last(fitness, self.path_dict['best'], self.path_dict['last'])
            # TODO maybe need a auto stop function for bad training

        self.model = self.release()
        coco_eval, _ = self.test_trained()

        # save coco results
        self.save_coco_results(coco_eval, self.path_dict['coco_results'])

        # release all
        self.close_tensorboard()
        self.release_cuda_cache()

    @torch.no_grad()
    def val_training(self):
        valer = self.val_class(model=self.model, half=True, dataloader=self.val_dataloader,
                               loss_fn=self.loss_fn, cls_names=self.datasets['names'],
                               epoch=self.epoch, writer=self.writer, visual_image=self.visual_image,
                               coco_json=self.coco_json, hyp=self.hyp)
        results = valer.val_training()
        return results

    # @torch.inference_mode()
    @torch.no_grad()
    def test_trained(self):
        self.epoch = -1
        self.checkpoint = self.load_checkpoint(self.path_dict['best'])
        if self.checkpoint is None:
            self.checkpoint = self.load_checkpoint(self.path_dict['last'])
            LOGGER.info('Load last.pt for validating because of no best.pt')
        else:
            LOGGER.info('Load best.pt for validating')

        self.model = self.load_model(load='model')

        tester = self.val_class(model=self.model, half=False, dataloader=self.test_dataloader,
                                loss_fn=self.loss_fn, cls_names=self.datasets['names'],
                                epoch=self.epoch, writer=self.writer, visual_image=self.visual_image,
                                coco_json=self.coco_json, hyp=self.hyp)
        results = tester.val_training()
        return results


def parse_args_detect(known: bool = False):
    r"""
    Parse args for training.
    Args:
        known: bool = True or False, Default=False.
            parser will get two namespace which the second is unknown args, if known=True.

    Returns:
        namespace(for setting args)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard', type=bool,
                        default=True, help='Use tensorboard to make visual')
    parser.add_argument('--visual_image', type=bool,
                        default=True, help='Make image (train val test) visual')
    parser.add_argument('--visual_graph', type=bool,
                        default=False, help='Make model graph visual')
    parser.add_argument('--weights', type=str,
                        default=str(ROOT / 'models/yolov5/yolov5s_v6.pt'), help='The path of checkpoint')
    # parser.add_argument('--weights', type=str, default='', help='The path of checkpoint')
    parser.add_argument('--freeze_names', type=list,
                        default=['backbone', 'neck'], help='Layer name to freeze in model')
    parser.add_argument('--device', type=str,
                        default='0', help='Use cpu or cuda:0 or 0')
    parser.add_argument('--epochs', type=int,
                        default=50, help='The epochs for training')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='The batch size in training')
    parser.add_argument('--workers', type=int,
                        default=0, help='For dataloader to load data')
    parser.add_argument('--shuffle', type=bool,
                        default=True, help='Shuffle the training data')
    parser.add_argument('--pin_memory', type=bool,
                        default=False, help='Load data to memory')
    parser.add_argument('--datasets', type=str,
                        default=str(ROOT / 'mine/data/datasets/detection/Customdatasets.yaml'),
                        help='The path of datasets.yaml')
    parser.add_argument('--save_name', type=str,
                        default='exp', help='The name of save dir')
    parser.add_argument('--save_path', type=str,
                        default=str(ROOT / 'runs/train/detect'), help='The save path of results')
    parser.add_argument('--hyp', type=str,
                        default=str(ROOT / 'data/hyp/hyp_detect_train.yaml'), help='The path of hyp.yaml')
    parser.add_argument('--augment', type=bool,
                        default=False, help='Use random augment image')
    parser.add_argument('--data_augment', type=str,
                        default='', help='The kind of data augmentation mosaic / mixup / cutout')
    parser.add_argument('--inc', type=int,
                        default=3, help='The image channel to input')
    parser.add_argument('--image_size', type=int,
                        default=640, help='The size of input image')
    parser.add_argument('--load_model', type=str,
                        default='state_dict', help="The pattern of loading model 'model' / 'state_dict' / None")
    parser.add_argument('--load_optimizer', type=bool,
                        default=False, help='True / False')
    parser.add_argument('--load_lr_scheduler', type=bool,
                        default=False, help='True / False')
    parser.add_argument('--load_gradscaler', type=bool,
                        default=False, help='True / False')
    parser.add_argument('--load_start_epoch', type=str,
                        default=None, help="The pattern of start training 'continue' / 'add' / None")
    parser.add_argument('--load_best_fitness', type=bool,
                        default=False, help='True / False')
    namespace = parser.parse_known_args()[0] if known else parser.parse_args()
    return namespace


@timer
def train_detection():
    arguments = parse_args_detect()
    trainer = TrainDetect(arguments)
    trainer.train()


if __name__ == '__main__':
    train_detection()
    import torch.optim.swa_utils

    # in the future
    # TODO add ema_model by torch.optim.swa_utils.AveragedModel
    # TODO colour str
    # TODO learn moviepy library sometimes

    # when need because it is complex
    # TODO add FLOPs compute module for model
    # TODO auto compute anchors

    # next work
    # TODO add necessary functions
    # TODO confusion matrix needed
    # TODO add plot curve functions for visual results
    # TODO design model structure
