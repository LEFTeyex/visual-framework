r"""
Mixin Module.
Consist of all Mixin class.
You can use the mixin module building train model(class), it is flexible, easy and quick.
The built example in train.py.

My rule of the design:
The Mixins only call self.* variable, can not define self.* variable inside to avoid confusion.
Please follow the rule, if you want to upgrade and maintain this module with me.
"""

import json
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from pycocotools.coco import COCO
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval

from utils import WRITER
from utils.log import LOGGER, add_log_file
from utils.check import check_only_one_set
from utils.bbox import xywh2xyxy, xyxy2x1y1wh, rescale_xyxy
from utils.general import delete_list_indices, time_sync, save_all_txt
from utils.decode import filter_outputs2predictions, non_max_suppression
from utils.metrics import match_pred_label_iou_vector, compute_metrics_per_class, compute_fitness
from utils.typeslib import _str_or_None, _pkt_or_None, _path, _dataset_c, _instance_c, \
    _module_or_None, _optimizer, _lr_scheduler, _gradscaler

__all__ = ['SetSavePathMixin', 'SaveCheckPointMixin', 'LoadAllCheckPointMixin', 'FreezeLayersMixin',
           'DataLoaderMixin', 'LossMixin', 'TrainDetectMixin', 'ValDetectMixin', 'COCOEvaluateMixin',
           'ResultsDealDetectMixin']


class SetSavePathMixin(object):
    r"""
    Need self.save_path, self.name.
    The function in the Mixin below.
    1. Set save path (then get them in dict) and add FileHandler for LOGGER. --- all self.*
    """

    def __init__(self):
        self.name = None
        self.save_path = None
        self.tensorboard = None

    def get_save_path(self, *args, logfile: _str_or_None = None):
        r"""
        Set save path (then get them in dict) and add FileHandler for LOGGER or add writer of tensorboard.
        Args:
            args: = (name, path), ...
            logfile: _str_or_None = 'logger.log' or '*/others.log', Default=None(logger.log file will not be created)

        Return dict{'name': strpath, ...}
        """
        self._set_name()
        self.save_path /= self.name
        self.save_path.mkdir(parents=True, exist_ok=True)

        # add FileHandler for LOGGER
        if logfile is not None:
            add_log_file(self.save_path / logfile)
        LOGGER.info('Setting save path...')

        # set tensorboard for self.writer
        if self.tensorboard:
            LOGGER.info('Setting tensorboard...')
            writer = WRITER.set_writer(self.save_path)
            LOGGER.info('Setting tensorboard successfully')
            LOGGER.info(f"See tensorboard results please run "
                        f"'tensorboard --logdir=runs/train/{self.name}/tensorboard' in Terminal")
        else:
            LOGGER.info('No tensorboard')
            writer = None

        # set save path for args(name, path) to dict
        save_dict = dict()
        for k, v in args:
            path = self.save_path / v
            path.parent.mkdir(parents=True, exist_ok=True)
            save_dict[k] = str(path)
        LOGGER.info('Set save path successfully')
        return save_dict, writer

    def _set_name(self):
        r"""Set name for save file without repeating, runs/*/exp1, runs/*/exp2"""
        self.save_path = Path(self.save_path)
        if self.save_path.exists():
            # to save number of exp
            list_num = []
            for dir_name in self.save_path.glob(f'{self.name}*'):  # get all dir matching (exp*)
                if dir_name.is_dir():
                    dir_name = str(dir_name.name).lstrip(self.name)
                    try:
                        list_num.append(int(dir_name))
                    except ValueError:
                        # no really error
                        LOGGER.error(f'The {dir_name} is an unexpected name '
                                     f'in end of directory that starts with exp')
            if list_num:
                self.name += str(max(list_num) + 1)  # set name, for example exp6
            else:
                self.name += '1'  # set name exp1

        else:
            self.name += '1'  # set name exp1


class SaveCheckPointMixin(object):
    def __init__(self):
        self.hyp = None
        self.model = None
        self.epoch = None
        self.epochs = None
        self.scaler = None
        self.save_dict = None
        self.optimizer = None
        self.lr_scheduler = None
        self.best_fitness = None
        self.checkpoint = None

    def get_checkpoint(self):
        LOGGER.debug('Getting checkpoint...')
        checkpoint = {'model': deepcopy(self.model),  # TODO de_parallel model in the future
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                      'gradscaler_state_dict': self.scaler.state_dict(),
                      'epoch': self.epoch,
                      'best_fitness': self.best_fitness
                      }
        LOGGER.debug('Get checkpoint successfully')
        return checkpoint

    def save_checkpoint(self, results):
        # save the best checkpoint
        fitness = compute_fitness(results, self.hyp['fit_weights'])  # compute fitness for best save
        if fitness > self.best_fitness:
            LOGGER.debug(f'Saving last checkpoint in epoch{self.epoch}...')
            checkpoint = self.get_checkpoint()
            self.best_fitness = fitness
            torch.save(checkpoint, self.save_dict['best'])
            LOGGER.debug(f'Save last checkpoint in epoch{self.epoch} successfully')

        # save the last checkpoint
        if self.epoch + 1 == self.epochs:
            LOGGER.info('Saving last checkpoint')
            checkpoint = self.get_checkpoint()
            torch.save(checkpoint, self.save_dict['last'])
            LOGGER.info('Save last checkpoint successfully')


class LoadAllCheckPointMixin(object):
    r"""
    Need self.model, self.device, self.checkpoint, self.param_groups, self.epochs
    The function in the Mixin below.
    1. Load checkpoint from path '*.pt' or '*.pth' file. ------------------ ues self.device
    2. Load model from 'model' or 'state_dict' or initialize model. ------- ues self.device, self.checkpoint
    3. Load optimizer from state_dict and add param_groups first. --------- ues self.checkpoint, self.param_groups
    4. Load lr_scheduler from state_dict. --------------------------------- ues self.checkpoint
    5. Load GradScaler from state_dict. ----------------------------------- ues self.checkpoint
    6. Load start_epoch. -------------------------------------------------- ues self.checkpoint, self.epochs
    7. Load best_fitness for choosing which weights of model is best among epochs. --- ues self.checkpoint
    8. Set param_groups, need to know how to set param_kind_list. --------- ues self.model
    """

    def __init__(self):
        self.model = None
        self.device = None
        self.epochs = None
        self.checkpoint = None
        self.param_groups = None

    def load_checkpoint(self, path: _path, suffix=('.pt', '.pth')):
        r"""
        Load checkpoint from path '*.pt' or '*.pth' file.
        Request:
            checkpoint = {'model':                    model,
                          'optimizer_state_dict':     optimizer.state_dict(),
                          'lr_scheduler_state_dict':  lr_scheduler.state_dict(),
                          'gradscaler_state_dict':    scaler.state_dict(),
                          'epoch':                    epoch,
                          'best_fitness':             best_fitness}
        Args:
            path: _path = Path
            suffix: tuple = ('.pt', '.pth', ...) the suffix of weight file, Default=('.pt', '.pth')

        Return checkpoint or None(when no file endwith the suffix)
        """
        LOGGER.info('Loading checkpoint...')
        path = Path(path)
        if path.is_file() and (path.suffix in suffix):
            # load checkpoint to self.device
            checkpoint = torch.load(path, map_location=self.device)
            LOGGER.info('Load checkpoint successfully')
            return checkpoint
        else:
            LOGGER.warning('There is no checkpoint to load. '
                           'Check the path, if checkpoint wanted.')
            return None

    def load_model(self, model_instance: _module_or_None = None, load: _str_or_None = 'state_dict'):
        r"""
        Load model from 'model' or 'state_dict' or initialize model.
        Args:
            model_instance: _module_or_None = ModelDetect() the instance of model, Default=None(only when load='model')
            load: _str_or_None = None / 'model' / 'state_dict', Default='state_dict'(load model from state_dict)

        Return model_instance in float
        """
        # check whether model_instance and load is conflict
        if load == 'model':
            if not check_only_one_set(model_instance, load):
                raise ValueError(f'Only one of {model_instance} '
                                 f'and {load} can be set, please reset them')
        elif model_instance is None:
            raise ValueError("The model_instance can not be None,"
                             " if load = None or 'state_dict'")

        if load is None:
            LOGGER.info('Initializing model weights...')
            model_instance.initialize_weights()
            LOGGER.info('Initialize model weights successfully')

        elif load == 'state_dict':
            LOGGER.info('Loading model state_dict...')
            self._check_checkpoint_not_none()
            state_dict = self.checkpoint['model'].state_dict()

            # delete the same keys but different weight shape
            model_ns = ((name, weight.shape) for name, weight in model_instance.state_dict().items())
            for name, shape in model_ns:
                if name in state_dict and state_dict[name].shape != shape:
                    del state_dict[name]
                    LOGGER.warning(f'Delete the {name} in state_dict because of different weight shape')

            # load model and get the rest of (0, missing_keys) and (1, unexpected_keys)
            rest = model_instance.load_state_dict(state_dict, strict=False)
            if rest[0] or rest[1]:
                LOGGER.warning(f'There are the rest of {len(rest[0])} missing_keys'
                               f' and {len(rest[1])} unexpected_keys when load model')
                LOGGER.warning(f'missing_keys: {rest[0]}')
                LOGGER.warning(f'unexpected_keys: {rest[1]}')
                LOGGER.info('Load model state_dict successfully with the rest of keys')
            else:
                LOGGER.info('Load model state_dict successfully without the rest of keys')

        elif load == 'model':
            LOGGER.info('Loading total model from checkpoint...')
            self._check_checkpoint_not_none()
            model_instance = self.checkpoint['model']
            LOGGER.info('Load total model successfully')

        else:
            raise ValueError(f"The arg load: {load} do not match, "
                             f"please input one of  (None, 'model', 'state_dict')")
        return model_instance.float().to(self.device)  # return model to self.device

    def load_optimizer(self, optim_instance: _optimizer, load: bool = True):
        r"""
        Load optimizer from state_dict and add param_groups first.
        Args:
            optim_instance: _optimizer = SGD(self.model.parameters(), lr=0.01) etc. the instance of optimizer
            load: bool = True / False, Default=True(load optimizer state_dict)

        Return optim_instance
        """
        LOGGER.info('Initialize optimizer successfully')

        if load:
            # load optimizer
            LOGGER.info('Loading optimizer state_dict...')
            self._check_checkpoint_not_none()
            optim_instance.load_state_dict(self.checkpoint['optimizer_state_dict'])
            LOGGER.info('Load optimizer state_dict successfully...')

        else:
            LOGGER.info('Do not load optimizer state_dict')
        return optim_instance

    def load_lr_scheduler(self, lr_scheduler_instance: _lr_scheduler, load: bool = True):
        r"""
        Load lr_scheduler from state_dict.
        Args:
            lr_scheduler_instance: _lr_scheduler = StepLR(self.optimizer, 30) etc. the instance of lr_scheduler
            load: bool = True / False, Default=True(load lr_scheduler state_dict)

        Return scheduler_instance
        """
        LOGGER.info('Initialize lr_scheduler successfully')
        if load:
            # load lr_scheduler
            LOGGER.info('Loading lr_scheduler state_dict...')
            self._check_checkpoint_not_none()
            lr_scheduler_instance.load_state_dict(self.checkpoint['lr_scheduler_state_dict'])
            LOGGER.info('Load lr_scheduler state_dict successfully...')

        else:
            LOGGER.info('Do not load lr_scheduler state_dict')
        return lr_scheduler_instance

    def load_gradscaler(self, gradscaler_instance: _gradscaler, load: bool = True):
        r"""
        Load GradScaler from state_dict.
        Args:
            gradscaler_instance: _gradscaler = GradScaler(enabled=self.cuda) etc. the instance of GradScaler
            load: bool = True / False, Default=True(load GradScaler state_dict)

        Return gradscaler_instance
        """
        LOGGER.info('Initialize GradScaler successfully')
        if load:
            # load GradScaler
            LOGGER.info('Loading GradScaler state_dict...')
            self._check_checkpoint_not_none()
            gradscaler_instance.load_state_dict(self.checkpoint['gradscaler_state_dict'])
            LOGGER.info('Load GradScaler state_dict successfully...')

        else:
            LOGGER.info('Do not load lr_scheduler state_dict')
        return gradscaler_instance

    def load_start_epoch(self, load: _str_or_None = 'continue'):
        r"""
        Load start_epoch.
        Args:
            load: _str_or_None = 'continue' / 'add' / None, Default='continue'
                continue: continue to train from start_epoch to self.epochs
                add: train self.epochs more times
                None: initialize start_epoch=0

        Return start_epoch
        """
        if load is None:
            # initialize start_epoch
            LOGGER.info('Initializing start_epoch...')
            start_epoch = 0
            LOGGER.info(f'Initialize start_epoch={start_epoch} successfully')
            LOGGER.info(f'The Model will be trained {self.epochs} times')

        elif load == 'continue':
            # load start_epoch to continue
            LOGGER.info('Loading start_epoch to continue...')
            self._check_checkpoint_not_none()
            start_epoch = self.checkpoint['epoch'] + 1
            if self.epochs < start_epoch:
                raise ValueError(f'The epochs: {self.epochs} can not be '
                                 f'less than the start_epoch: {start_epoch}')
            LOGGER.info(f'Load start_epoch={start_epoch} to continue successfully')
            LOGGER.info(f'The Model will be trained {self.epochs - start_epoch + 1} times')

        elif load == 'add':
            # load start_epoch to add epochs to train
            LOGGER.info('Loading start_epoch to add epochs to train...')
            self._check_checkpoint_not_none()
            start_epoch = self.checkpoint['epoch'] + 1
            self.epochs += self.checkpoint['epoch']
            LOGGER.info(f'Load start_epoch={start_epoch} to add epochs to train successfully')
            LOGGER.info(f'The Model will be trained {self.epochs} times')

        else:
            raise ValueError(f"The arg load: {load} do not match, "
                             f"please input one of  (None, 'continue', 'add')")
        return start_epoch

    def load_best_fitness(self, load: bool = True):
        r"""
        Load best_fitness for choosing which weights of model is best among epochs.
        Args:
            load: bool = True / False, Default=True(load best_fitness)

        Return best_fitness
        """
        if load:
            # load best_fitness
            LOGGER.info('Loading best_fitness...')
            self._check_checkpoint_not_none()
            best_fitness = self.checkpoint['best_fitness']
            LOGGER.info('Load best_fitness successfully')

        else:
            # initialize best_fitness
            LOGGER.info('Initializing best_fitness...')
            # todo args can change
            best_fitness = 0.0
            LOGGER.info(f'Initialize best_fitness={best_fitness} successfully')
        return best_fitness

    def set_param_groups(self, param_kind_tuple: _pkt_or_None = None):
        r"""
        Set param_groups, need to know how to set param_kind_list.
        The parameters in param_groups will not repeat.
        Args:
            param_kind_tuple: _pkt_or_None =    (('bias', nn.Parameter, {lr=0.01}),
                                                 ('weight', nn.BatchNorm2d, {lr=0.02}))
                                                Default=None(do not set param_groups)
                                   for example: (('weight'/'bias', nn.Parameter/nn.*, {lr=0.01, ...}), ...)

        Returns:
            param_groups [{'params': nn.Parameter, 'lr': 0.01, ...}, ...]
        """
        # do not set param_groups
        if param_kind_tuple is None:
            LOGGER.info('No param_groups set')
            return None
        # set param_groups
        LOGGER.info('Setting param_groups...')
        # TODO Upgrade the algorithm in the future for filtering parameters from model.modules() better
        param_groups = []
        rest = []  # save_params the rest parameters that are not filtered
        indices = []  # save_params parameters (index, name) temporarily to delete

        # get all parameters from model and set one of ('weightbias', 'weight', 'bias') to its indices
        for element in self.model.modules():
            # todo args can change
            if hasattr(element, 'weight') or hasattr(element, 'bias'):
                rest.append(element)
                if element.weight is not None:
                    indices.append('weight')
                    if element.bias is not None:
                        indices[-1] += 'bias'
                elif element.bias is not None:
                    indices.append('bias')

        # loop the param_groups
        for name, kind, param_dict in param_kind_tuple:
            str_eval = 'module.' + name  # for eval to get code module.weight or module.bias
            save_params = []  # save parameters temporarily

            # filter parameters from model
            for index, module in enumerate(rest):
                if hasattr(module, name):  # has module.weight / module.bias
                    module_name = eval(str_eval)  # module.weight / module.bias

                    if isinstance(module, kind):
                        indices[index] = indices[index].replace(name, '')
                        save_params.append(module_name)

                    elif isinstance(module_name, kind):
                        indices[index] = indices[index].replace(name, '')
                        save_params.append(module_name)

                    elif isinstance(module, nn.Module):
                        # let nn.others pass to prevent other unexpected problems from happening
                        pass

                    else:
                        raise Exception(f'Unexpected error, please check code carefully,'
                                        f' especially for {name}, {kind}')

            # remove the element (that got (weight and bias) for param_groups) in the rest and its msg in indices
            indices_filter = []
            for index, element in enumerate(indices):
                # filter right index
                if not element:  # if empty
                    indices_filter.append(index)

            delete_list_indices(indices, indices_filter)
            delete_list_indices(rest, indices_filter)

            # add params to dict and add param_dict to param_groups
            param_dict['params'] = save_params
            param_dict['name'] = f'{kind.__name__}.{name}'
            param_groups.append(param_dict)

        # check whether rest is empty
        if rest:
            raise Exception(f'There are still {len(rest)} modules ungrouped,'
                            f' please reset the arg param_kind_tuple')
        LOGGER.info('Set param_groups successfully')
        return param_groups

    def _check_checkpoint_not_none(self):
        r"""Check whether self.checkpoint exists"""
        if self.checkpoint is None:
            raise ValueError('The self.checkpoint is None, please load checkpoint')


class FreezeLayersMixin(object):
    def __init__(self):
        self.model = None

    def freeze_layers(self, layer_names: list):
        r"""
        Freeze layers in model by names.
        Args:
            layer_names: list = list consist of name in model layers
        """
        if layer_names:
            LOGGER.info(f'Freezing name {layer_names} in model...')
            # to string
            for idx, name in enumerate(layer_names):
                layer_names[idx] = str(name)

            for name, param in self.model.named_parameters():
                if any(x in name for x in layer_names):
                    param.requires_grad = False
                    LOGGER.info(f'Frozen {name} layer')
            LOGGER.info('Frozen successfully')

    def unfreeze_layers(self, layer_names: list):
        r"""
        Unfreeze layers in model by names.
        Args:
            layer_names: list = list consist of name in model layers
        """
        if layer_names:
            LOGGER.info(f'Unfreezing name {layer_names} in model...')
            # to string
            for idx, name in enumerate(layer_names):
                layer_names[idx] = str(name)

            for name, param in self.model.named_parameters():
                if any(x in name for x in layer_names):
                    param.requires_grad = True
                    LOGGER.info(f'Unfrozen {name} layer')
            LOGGER.info('Unfrozen successfully')

    def freeze_model(self):
        r"""Freeze model totally"""
        LOGGER.info('Freezing all layers of model...')
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            LOGGER.debug(f'Frozen {name}')
        LOGGER.info('Frozen all layers of model successfully')

    def unfreeze_model(self):
        r"""Unfreeze model totally"""
        LOGGER.info('Unfreezing all layers of model...')
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            LOGGER.debug(f'Unfrozen {name}')
        LOGGER.info('Unfrozen all layers of model successfully')


class DataLoaderMixin(object):
    r"""
    Need self.hyp, self.datasets, self.image_size, self.batch_size, self.workers, self.pin_memory
    The function in the Mixin below.
    1. Set dataset and get dataloader. --- all self.*
    """

    def __init__(self):
        self.hyp = None
        self.writer = None
        self.workers = None
        self.datasets = None
        self.image_size = None
        self.batch_size = None
        self.pin_memory = None
        self.visual_image = None

    def get_dataloader(self, dataset: _dataset_c, name: str, augment: bool = False,
                       data_augment='', shuffle: bool = False, create_json_gt=None):
        r"""
        Set dataset and get dataloader.
        Args:
            dataset: _dataset = dataset class
            name: str = 'train' / 'val' / 'test'
            augment: bool = False/True
            data_augment: str = 'cutout'/'mixup'/'mosaic' the kind of data augmentation
            shuffle: bool = False/True
            create_json_gt: = path for saving json whether create json coco format for using COCOeval

        Return dataloader instance
        """
        if self.datasets.get(name, None) is None:
            dataloader = None
            LOGGER.warning(f'The datasets {name} do not exist, got Dataloader {name} is None')
            if create_json_gt:
                raise Exception(f'The json_gt will not be created.'
                                f'Please check and correct the path of {name} in datasets.yaml, '
                                f'if you want to use COCO to eval.'
                                f'Otherwise please do not input the arg of the create_json_gt')
        else:
            LOGGER.info(f'Initializing Dataloader {name}...')
            # set dataset
            LOGGER.info(f'Initializing Dataset {name}...')
            dataset = dataset(self.datasets, name, self.image_size, augment, data_augment, self.hyp,
                              create_json_gt=create_json_gt)
            LOGGER.info(f'Initialize Dataset {name} successfully')

            # visualizing
            if self.visual_image:
                LOGGER.info(f'Visualizing Dataset {name}...')
                WRITER.add_datasets_images_labels_detect(self.writer, dataset, name)
                LOGGER.info(f'Visualize Dataset {name} successfully')

            # set dataloader
            # TODO upgrade num_workers(deal and check) and sampler(distributed.DistributedSampler) for DDP
            batch_size = min(self.batch_size, len(dataset))
            dataloader = DataLoader(dataset, batch_size, shuffle,
                                    num_workers=self.workers,
                                    pin_memory=self.pin_memory,
                                    collate_fn=dataset.collate_fn)
            LOGGER.info(f'Initialize Dataloader {name} successfully')
        return dataloader


class LossMixin(object):
    r"""
    Need self.hyp, self.model
    The function in the Mixin below.
    1. Get loss function. --- all self.*
    """

    def __init__(self):
        self.hyp = None
        self.model = None

    def get_loss_fn(self, loss_class: _instance_c):
        r"""
        Get loss function.
        Args:
            loss_class: _instance_c = loss class

        Returns:
            loss_instance
        """
        LOGGER.info('Initializing LossDetect...')
        loss_instance = loss_class(self.model, self.hyp)
        LOGGER.info('Initialize LossDetect successfully')
        return loss_instance


class TrainDetectMixin(object):
    def __init__(self):
        self.inc = None
        self.cuda = None
        self.epoch = None
        self.model = None
        self.epochs = None
        self.device = None
        self.scaler = None
        self.writer = None
        self.loss_fn = None
        self.optimizer = None
        self.image_size = None
        self.visual_graph = None
        self.lr_scheduler = None
        self.train_dataloader = None

    def train_one_epoch(self):
        self.model.train()
        if self.visual_graph:
            WRITER.add_model_graph(self.writer, self.model, self.inc, self.image_size, self.epoch)
        loss_name = ('total_loss', 'bbox_loss', 'class_loss', 'object_loss')
        loss_all_mean = torch.zeros(4, device=self.device)

        with tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader),
                  bar_format='{l_bar}{bar:10}{r_bar}') as pbar:
            for index, (images, labels, _, _) in pbar:
                images = images.to(self.device).float() / 255  # to float32 and normalized 0.0-1.0
                labels = labels.to(self.device)

                # TODO warmup in the future

                # forward with mixed precision
                with autocast(enabled=self.cuda):
                    outputs = self.model(images)
                    loss, loss_items = self.loss_fn(outputs, labels, kind='ciou')

                # backward and optimize
                if self.scaler is None:
                    self.optimize_no_scale(loss)
                else:
                    self.optimize_scale(loss)

                # mean total loss and loss items
                loss_all = torch.cat((loss, loss_items), dim=0).detach()
                loss_all_mean = self.loss_mean(index, loss_all_mean, loss_all)
                self._show_loss_in_pbar_training(loss_all_mean, pbar)

            WRITER.add_optimizer_lr(self.writer, self.optimizer, self.epoch)
            WRITER.add_epoch_curve(self.writer, 'train_loss', loss_all_mean, loss_name, self.epoch)

            # lr_scheduler
            self.lr_scheduler.step()
        return loss_all_mean.tolist(), loss_name

    def optimize_no_scale(self, loss):
        # backward
        loss.backward()

        # optimize
        self.optimizer.step()
        self.optimizer.zero_grad()

    def optimize_scale(self, loss):
        # backward
        self.scaler.scale(loss).backward()

        # optimize
        # todo maybe accumulating gradient will be better (if index...: ...)
        self.scaler.step(self.optimizer)  # optimizer.step()
        self.scaler.update()
        self.optimizer.zero_grad()  # improve a little performance when set_to_none=True

    def _show_loss_in_pbar_training(self, loss, pbar):
        # GPU memory used which an approximate value because of 1E9
        memory_cuda = f'GPU: {torch.cuda.memory_reserved(self.device) / 1E9 if torch.cuda.is_available() else 0:.3f}GB'

        # GPU memory used which an accurate value because of 1024 * 1024 * 1024 = 1073741824 but slow
        # memory_cuda = f'GPU: {torch.cuda.memory_reserved() / 1073741824 if torch.cuda.is_available() else 0:.3f}GB'

        # show in pbar
        space = ' ' * 11
        progress = f'{self.epoch}/{self.epochs - 1}:'
        pbar.set_description_str(f"{space}epoch {progress:<9}{memory_cuda}")
        pbar.set_postfix_str(f'total_loss: {loss[0]:.3f}, '
                             f'bbox_loss: {loss[1]:.3f}, '
                             f'class_loss: {loss[2]:.3f}, '
                             f'object_loss: {loss[3]:.3f}')

    @staticmethod
    def loss_mean(index, loss_all_mean, loss_all):
        loss_all_mean = (loss_all_mean * index + loss_all) / (index + 1)
        return loss_all_mean


class EMAModelMixin(object):
    def get_ema_model(self):
        raise NotImplementedError


class DDPModelMixin(object):
    def get_ddp_model(self):
        raise NotImplementedError


class SyncBatchNormMixin(object):
    def get_sync_bn_model(self):
        raise NotImplementedError


class CheckMixin(object):
    def check_batch_size(self):
        raise NotImplementedError

    def check_image_size(self):
        raise NotImplementedError


class ValDetectMixin(object):
    def __init__(self):
        self.time = None
        self.seen = None
        self.half = None
        self.model = None
        self.epoch = None
        self.writer = None
        self.device = None
        self.loss_fn = None
        self.coco_eval = None
        self.dataloader = None
        self.visual_image = None

    def val_once(self):
        loss_name = ('total_loss', 'bbox_loss', 'class_oss', 'object_loss')
        loss_all_mean = torch.zeros(4, device=self.device).half() if self.half else torch.zeros(4, device=self.device)
        iou_vector = torch.linspace(0.5, 0.95, 10, device=self.device)
        stats = []  # statistics to save tuple(pred_iou_level, pred conf, pred cls, label cls)
        json_dt = []  # save detection truth for COCO
        with tqdm(enumerate(self.dataloader), total=len(self.dataloader),
                  bar_format='{l_bar:>42}{bar:10}{r_bar}') as pbar:
            for index, (images, labels, shape_converts, img_ids) in pbar:
                # get current for computing FPs but maybe not accurate maybe
                t0 = time_sync()
                images = images.to(self.device)
                labels = labels.to(self.device)

                # to half16 or float32 and normalized 0.0-1.0
                images = images.half() / 255 if self.half else images.float() / 255
                bs, _, h, w = images.shape

                # inference
                outputs = self.model(images)
                loss, loss_items = self.loss_fn(outputs, labels, kind='ciou')

                # mean total loss and loss items
                loss_all = torch.cat((loss, loss_items), dim=0).detach()
                loss_all_mean = TrainDetectMixin.loss_mean(index, loss_all_mean, loss_all)
                self._show_loss_in_pbar_validating(loss_all_mean, pbar)

                # convert labels
                labels[:, 2:] *= torch.tensor([w, h, w, h], device=self.device)  # pixel scale

                # parse outputs to predictions
                predictions = self._parse_outputs(outputs)

                self.time += time_sync() - t0

                # nms
                predictions = non_max_suppression(predictions, 0.5, 300)  # list which len is batch size

                if self.visual_image:
                    WRITER.add_batch_images_predictions_detect(self.writer, 'test_pred', index, images, predictions,
                                                               self.epoch)

                # get metrics data
                self._get_metrics_stats(predictions, labels, shape_converts, iou_vector, stats, json_dt, img_ids)

            WRITER.add_epoch_curve(self.writer, 'val_loss', loss_all_mean, loss_name, self.epoch)

        if self.coco_eval:
            with open(self.coco_eval[1], 'w') as f:
                json.dump(json_dt, f)
        return loss_all_mean.tolist(), loss_name, stats

    def compute_metrics(self, stats):
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # all of pred_iou_level, pred cls_conf, pred cls, label cls
        # cls count
        cls_number = np.bincount(stats[3].astype(np.int64), minlength=self.model.nc)
        cls_number.tolist()
        fmt = '<10.3f'
        space = ' ' * 50

        if stats and stats[0].any():
            # consist of all IoU 0.50:0.95 metrics
            ap, f1, p, r, cls = compute_metrics_per_class(*stats)
            cls = cls.tolist()

            # deal metrics
            ap_mean, f1_mean, p_mean, r_mean = deepcopy(ap), deepcopy(f1), deepcopy(p), deepcopy(r)
            ap50, ap75, ap50_95 = ap_mean[:, 0].mean(), ap_mean[:, 5].mean(), ap_mean.mean(axis=1).mean()
            mf1, mp, mr = f1_mean.mean(axis=0), p_mean.mean(axis=0), r_mean.mean(axis=0)

            # return
            ap_all = (ap50_95.tolist(), ap50.tolist(), ap75.tolist(), ap.tolist())
            f1_all = (mf1.tolist(), f1.tolist())
            p_all = (mp.tolist(), p.tolist())
            r_all = (mr.tolist(), r.tolist())
            # the F1, P, R when IoU=0.50
            LOGGER.info(f'{space}P_50: {mp[0]:{fmt}}'
                        f'R_50: {mr[0]:{fmt}}'
                        f'F1_50: {mf1[0]:{fmt}}'
                        f'AP50: {ap50:{fmt}}'
                        f'AP75: {ap75:{fmt}}'
                        f'AP/AP5095: {ap50_95:{fmt}}')
        else:
            ap_all, f1_all, p_all, r_all, cls = ((None,) * 4), ((None,) * 2), ((None,) * 2), ((None,) * 2), None
            LOGGER.info(f'{space}P_50: {0:{fmt}}'
                        f'R_50: {0:{fmt}}'
                        f'F1_50: {0:{fmt}}'
                        f'AP50: {0:{fmt}}'
                        f'AP75: {0:{fmt}}'
                        f'AP/AP5095: {0:{fmt}}')

        cls_name_number = (cls, cls_number)
        return ap_all, f1_all, p_all, r_all, cls_name_number

    def _parse_outputs(self, outputs):
        outputs = self.model.decode_outputs(outputs, self.model.scalings)  # bbox is xyxy
        # TODO get it by predictions or outputs through below whether used outputs over
        # TODO think it in memory for filter_outputs2predictions
        outputs = filter_outputs2predictions(outputs, 0.25)  # list which len is bs
        return outputs

    def _get_metrics_stats(self, predictions, labels, shape_converts, iou_vector, stats, json_dt, img_ids):
        # save tuple(pred_iou_level, pred conf, pred cls, label cls)
        for index, (pred, image_id) in enumerate(zip(predictions, img_ids)):
            self.seen += 1
            label = labels[labels[:, 0] == index, 1:]  # one image label
            nl = label.shape[0]  # number of label
            label_cls = label[:, 0].long().tolist() if nl else []

            if not pred.shape[0]:
                if nl:
                    pred_iou_level = torch.zeros((0, iou_vector.shape[0]), dtype=torch.bool)
                    stats.append((pred_iou_level, torch.Tensor(), torch.Tensor(), label_cls))
                continue

            pred[:, :4] = rescale_xyxy(pred[:, :4], shape_converts[index])

            if self.coco_eval:
                self._append_json_dt(pred, image_id, json_dt)

            if nl:
                label[:, 1:] = xywh2xyxy(label[:, 1:])
                label[:, 1:] = rescale_xyxy(label[:, 1:], shape_converts[index])
                pred_iou_level = match_pred_label_iou_vector(pred, label, iou_vector)
                # TODO confusion matrix needed
            else:
                pred_iou_level = torch.zeros((pred.shape[0], iou_vector.shape[0]), dtype=torch.bool)
            stats.append((pred_iou_level.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), label_cls))

    @staticmethod
    def _append_json_dt(predictions, image_id, json_dt: list):
        # prediction(n, 6(x, y, x, y, conf, cls_idx))
        bboxes = xyxy2x1y1wh(predictions[:, :4])
        cls = predictions[:, 5]
        scores = predictions[:, 4]
        for category_id, bbox, score in zip(cls.tolist(), bboxes.tolist(), scores.tolist()):
            json_dt.append({'image_id': int(image_id),
                            'category_id': int(category_id),
                            'bbox': [round(x, 3) for x in bbox],  # Keep 3 decimal places to get more accurate mAP
                            'score': float(score)})

    def _show_loss_in_pbar_validating(self, loss, pbar):
        # GPU memory used which an approximate value because of 1E9
        memory_cuda = f'GPU: {torch.cuda.memory_reserved(self.device) / 1E9 if torch.cuda.is_available() else 0:.3f}GB'

        # GPU memory used which an accurate value because of 1024 * 1024 * 1024 = 1073741824 but slow
        # memory_cuda = f'GPU: {torch.cuda.memory_reserved() / 1073741824 if torch.cuda.is_available() else 0:.3f}GB'

        # show in pbar
        space = ' ' * 11
        pbar.set_description_str(f"{space}{'validating:':<15}{memory_cuda}")
        pbar.set_postfix_str(f'total_loss: {loss[0]:.3f}, '
                             f'bbox_loss: {loss[1]:.3f}, '
                             f'class_loss: {loss[2]:.3f}, '
                             f'object_loss: {loss[3]:.3f}')


class COCOEvaluateMixin(object):
    def __init__(self):
        self.coco_eval = None

    def coco_evaluate(self, dataloader, eval_type='bbox'):  # eval_type is one of ('segm', 'bbox', 'keypoints')
        coco_gt, coco_dt = self.coco_eval
        if not Path(coco_gt).exists():
            raise FileExistsError(f'The coco_gt {coco_gt} json file do not exist')
        if not Path(coco_dt).exists():
            raise FileExistsError(f'The coco_dt {coco_dt} json file do not exist')

        coco_gt = COCO(coco_gt)
        coco_dt = coco_gt.loadRes(coco_dt)
        coco = COCOeval(coco_gt, coco_dt, iouType=eval_type)
        coco.params.imgIds = list(dataloader.dataset.indices)
        coco.evaluate()
        coco.accumulate()
        coco.summarize()
        return coco.eval

    def save_coco_results(self, coco_results, path):
        path = str(path)
        if self.coco_eval:
            params = coco_results['params']
            new_params = {}
            for k, v in params.__dict__.items():
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                elif isinstance(v, (list, tuple)):
                    v = np.asarray(v).tolist()
                new_params[k] = v
            coco_results['params'] = new_params

            for k, v in coco_results.items():
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                elif isinstance(v, datetime):
                    v = str(v)
                coco_results[k] = v

            with open(path, 'w') as f:
                json.dump(coco_results, f)
                LOGGER.info(f"Save json coco_dt {path} successfully")


class ResultsDealDetectMixin(object):
    def __init__(self):
        self.epoch = None
        self.results = None
        self.datasets = None
        self.save_dict = None

        # itself
        self.saved = False

    def get_results_dict(self, *args: tuple):
        r"""
        Get dict of results for saving.
        Args:
            *args: (name, type to save), if None get default setting

        Return results dict(for self.results)
        """
        if not args:
            args = self._get_results_path_type()
        results = {}
        for name, value in args:
            results[name] = value
        return results

    @staticmethod
    def _get_results_path_type():
        return ('all_results', []), ('all_class_results', [])

    def add_results_dict(self, *args: tuple):
        self._check_results_exists_dict()

        for name, value in args:
            self.results[name] = value

    def add_data_results(self, *arg: tuple):
        r"""
        Add data to self.results in name.
        Args:
            *arg: tuple = (name, data(tuple or list))
        """
        self._check_results_exists_dict()
        for name, data in arg:
            self.results[name].extend(data)

    def deal_results_memory(self, results_train, results_val, max_row: int = 1024):
        self.saved = False
        train_loss = results_train
        val_loss, metrics, fps_time = results_val
        (ap50_95, ap50, ap75, ap), (mf1, f1), (mp, p), (mr, r), (cls, cls_number) = metrics
        all_results = []
        all_class_results = []

        # train_val
        if self.epoch == 0:
            title_train_val = ['epoch', 'train:', train_loss[1], 'val:', val_loss[1],
                               'ap50', 'ap75', 'ap50_95', 'mf1', 'mp', 'mr']
            all_results.append(title_train_val)
        data_train_val = [self.epoch, train_loss[0], val_loss[0], ap50, ap75, ap50_95, mf1, mp, mr]
        all_results.append(data_train_val)

        # all class
        title_all_class = [f'epoch{self.epoch}', 'cls_name', 'cls_number', '(IoU=0.50:0.95):', 'AP', 'F1', 'P', 'R']
        if ap is not None:
            rest = (cls, ap, f1, p, r)
            all_class_results.append(title_all_class)
            for c, ap_c, f1_c, p_c, r_c in zip(*rest):
                name_c = self.datasets['names'][c]
                number_c = cls_number[c]
                data_all_class = [name_c, number_c, ap_c, f1_c, p_c, r_c]
                all_class_results.append(data_all_class)
        else:
            nc = cls_number.shape[0]
            all_class_results.append(title_all_class)
            for c in range(nc):
                name_c = self.datasets['names'][c]
                number_c = cls_number[c]
                data_all_class = [name_c, number_c, None, None, None, None]
                all_class_results.append(data_all_class)

        # auto save
        self._auto_save(max_row)

        # return for saving and computing best_fitness
        return (all_results, all_class_results), \
               (mp[0], mr[0], mf1[0], ap50, ap50_95) if ap50 is not None else (0,) * 5

    def save_all_results(self):
        r"""Save all content in results then empty self.results"""
        if self.saved:
            return None
        self._check_results_exists_dict()
        to_save = []
        for k, v in self.results.items():
            to_save.append((v, self.save_dict[k]))
        save_all_txt(*to_save, mode='a')
        for key in self.results:
            self.results[key] = []

    def _auto_save(self, max_row):
        r"""Auto save self.results if any len(list) is more than max_row"""
        for v in self.results.values():
            if len(v) > max_row:
                self.save_all_results()
                self.saved = True
                break

    def _check_results_exists_dict(self):
        r"""Check whether self.results exists"""
        if not self.results:
            raise ValueError('The self.results do not exist, please set it')
        assert isinstance(self.results, dict), f'Except the type of self.results is dict but got {type(self.results)}'


if __name__ == '__main__':
    pass
