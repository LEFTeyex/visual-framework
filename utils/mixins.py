r"""
Mixin Module.
Consist of all Mixin class.
You can use the mixin module building train model(class), it is flexible, easy and quick.
The built example in train.py.

My rule of the design:
The Mixins only call self.* variable, can not define self.* variable inside to avoid confusion.
Please follow the rule, if you want to upgrade and maintain this module with me.
"""

import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from copy import deepcopy
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from utils.log import LOGGER, add_log_file
from utils.check import check_only_one_set
from utils.general import delete_list_indices, time_sync, save_all_txt
from utils.bbox import xywh2xyxy, rescale_xyxy
from utils.decode import parse_outputs_yolov5, filter_outputs2predictions, non_max_suppression
from utils.metrics import match_pred_label_iou_vector, compute_metrics_per_class, compute_fitness
from utils.typeslib import \
    _str_or_None, \
    _module_or_None, _optimizer, _lr_scheduler, _gradscaler, \
    _dataset_c, _instance_c, \
    _pkt_or_None

__all__ = ['SetSavePathMixin', 'SaveCheckPointMixin', 'LoadAllCheckPointMixin', 'DataLoaderMixin', 'LossMixin',
           'TrainDetectMixin', 'ValDetectMixin', 'ResultsDealDetectMixin']


class SetSavePathMixin(object):
    r"""
    Need self.save_path, self.name.
    The function in the Mixin below.
    1. Set save path (then get them in dict) and add FileHandler for LOGGER. --- all self.*
    """

    def __init__(self):
        self.name = None
        self.save_path = None

    def get_save_path(self, *args, logfile: _str_or_None = None):
        r"""
        Set save path (then get them in dict) and add FileHandler for LOGGER.
        Args:
            args: = (name, path), ...
            logfile: _str_or_None = 'logger.log' or '*/others.log', Default=None(logger.log file will not be created)

        Return dict{'name': path, ...}
        """
        self._set_name()
        self.save_path /= self.name
        self.save_path.mkdir(parents=True, exist_ok=True)

        # add FileHandler for LOGGER
        if logfile is not None:
            add_log_file(self.save_path / logfile)
        LOGGER.info('Setting save path...')

        # set save path for args(name, path) to dict
        save_dict = dict()
        for k, v in args:
            path = self.save_path / v
            path.parent.mkdir(parents=True, exist_ok=True)
            save_dict[k] = path
        LOGGER.info('Set save path successfully')
        return save_dict

    def _set_name(self):
        r"""Set name for save file without repeating, runs/*/464, runs/*/456456"""
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
        checkpoint = {'model': self.model,  # TODO de_parallel model in the future
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
    Need self.model, self.device, self.weights, self.checkpoint, self.param_groups, self.epochs
    The function in the Mixin below.
    1. Load checkpoint from self.weights '*.pt' or '*.pth' file. ---------- ues self.device, self.weights
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
        self.weights = None
        self.checkpoint = None
        self.param_groups = None

    def load_checkpoint(self, suffix=('.pt', '.pth')):
        r"""
        Load checkpoint from self.weights '*.pt' or '*.pth' file.
        Request:
            checkpoint = {'model':                    model,
                          'optimizer_state_dict':     optimizer.state_dict(),
                          'lr_scheduler_state_dict':  lr_scheduler.state_dict(),
                          'gradscaler_state_dict':    scaler.state_dict(),
                          'epoch':                    epoch,
                          'best_fitness':             best_fitness}
        Args:
            suffix: tuple = ('.pt', '.pth', ...) the suffix of weight file, Default=('.pt', '.pth')

        Return checkpoint or None(when no file endwith the suffix)
        """
        LOGGER.info('Loading checkpoint...')
        if self.weights.is_file() and (self.weights.suffix in suffix):
            # load checkpoint to self.device
            checkpoint = torch.load(self.weights, map_location=self.device)
            LOGGER.info('Load checkpoint successfully')
            return checkpoint
        else:
            LOGGER.warning('There is no checkpoint to load. '
                           'Check the weights args, if checkpoint wanted.')
            return None

    def load_model(self, model_instance: _module_or_None = None, load: _str_or_None = 'state_dict'):
        r"""
        Load model from 'model' or 'state_dict' or initialize model.
        Args:
            model_instance: _module_or_None = ModelDetect() the instance of model, Default=None(only when load='model')
            load: _str_or_None = None / 'model' / 'state_dict', Default='state_dict'(load model from state_dict)

        Return model_instance
        """
        # check whether model_instance and load is conflict
        if load == 'model' and check_only_one_set(model_instance, load):
            raise ValueError(f'Only one of {model_instance} '
                             f'and {load} can be set, please reset them')
        else:
            if model_instance is None:
                raise ValueError("The model_instance can not be None,"
                                 " if load = None or 'state_dict'")

        if load is None:
            # TODO: initialize model weights by new way
            pass

        elif load == 'state_dict':
            LOGGER.info('Loading model state_dict...')
            self._check_checkpoint_not_none()
            # TODO: Upgrade for somewhere in the future for Transfer Learning
            # load model and get the rest of (0, missing_keys) and (1, unexpected_keys)
            rest = model_instance.load_state_dict(self.checkpoint['model'].state_dict(), strict=False)
            if rest[0] or rest[1]:
                LOGGER.warning(f'There are the rest of {len(rest[0])} missing_keys'
                               f' and {len(rest[1])} unexpected_keys when load model')
                LOGGER.info('Load model state_dict successfully with the rest of keys')
            else:
                LOGGER.info('Load model state_dict successfully without the rest of keys')

        elif load == 'model':
            LOGGER.info('Loading total model from checkpoint...')
            self._check_checkpoint_not_none()
            model_instance = self.checkpoint.get['model']
            LOGGER.info('Load total model successfully')

        else:
            raise ValueError(f"The arg load: {load} do not match, "
                             f"please input one of  (None, 'model', 'state_dict')")
        return model_instance.to(self.device)  # return model to self.device

    def load_optimizer(self, optim_instance: _optimizer, load: bool = True):
        r"""
        Load optimizer from state_dict and add param_groups first.
        Args:
            optim_instance: _optimizer = SGD(self.model.parameters(), lr=0.01) etc. the instance of optimizer
            load: bool = True / False, Default=True(load optimizer state_dict)

        Return optim_instance
        """
        LOGGER.info('Initialize optimizer successfully')
        # add param_groups
        LOGGER.info('Adding param_groups...')
        if self.param_groups is None:
            LOGGER.info('Add no additional param_groups')

        elif isinstance(self.param_groups, list):
            if not self.param_groups:
                LOGGER.info('Add no additional param_groups ')
            else:
                for param_group in self.param_groups:
                    optim_instance.add_param_group(param_group)
                LOGGER.info('Add param_groups successfully')

        else:
            raise TypeError(f'The self.param_groups: '
                            f'{type(self.param_groups)} must be list or None')

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
            # todo: args can change
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

        Return param_groups {'params': nn.Parameter, 'lr': 0.01, ...}
        """
        # do not set param_groups
        if param_kind_tuple is None:
            LOGGER.info('No param_groups set')
            return None
        # set param_groups
        LOGGER.info('Setting param_groups...')
        # TODO: Upgrade the algorithm in the future for filtering parameters from model.modules() better
        param_groups = []
        rest = []  # save_params the rest parameters that are not filtered
        indices = []  # save_params parameters (index, name) temporarily to delete

        # get all parameters from model and set one of ('weightbias', 'weight', 'bias') to its indices
        for element in self.model.modules():
            # todo: args can change
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

            indices = delete_list_indices(indices, indices_filter)
            rest = delete_list_indices(rest, indices_filter)

            # add params to dict and add param_dict to param_groups
            param_dict['params'] = save_params
            param_groups.append(param_dict)

        # check whether rest is empty
        if rest:
            raise Exception(f'There are still {len(rest)} modules ungrouped,'
                            f' please reset the arg param_kind_tuple')
        LOGGER.info('Set param_groups successfully')
        return param_groups

    def _check_checkpoint_not_none(self):
        r"""Check whether self.checkpoint exists"""
        if not self.checkpoint:
            raise ValueError('The self.checkpoint is None, please load checkpoint')


class DataLoaderMixin(object):
    r"""
    Need self.datasets, self.image_size, self.batch_size, self.shuffle, self.workers, self.pin_memory
    The function in the Mixin below.
    1. Set dataset and get dataloader. --- all self.*
    """

    def __init__(self):
        self.shuffle = None
        self.workers = None
        self.datasets = None
        self.image_size = None
        self.batch_size = None
        self.pin_memory = None

    def get_dataloader(self, dataset: _dataset_c, name: str):
        r"""
        Set dataset and get dataloader.
        Args:
            dataset: _dataset = dataset class
            name: str = 'train' / 'val' / 'test'

        Return dataloader instance
        """
        LOGGER.info('Initializing Dataloader...')
        # set dataset
        LOGGER.info('Initializing Dataset...')
        dataset = dataset(self.datasets[name], self.image_size, name)
        LOGGER.info('Initialize Dataset successfully')

        # set dataloader
        # TODO upgrade num_workers(deal and check) and sampler(distributed.DistributedSampler) for DDP
        batch_size = min(self.batch_size, len(dataset))
        dataloader = DataLoader(dataset, batch_size, self.shuffle,
                                num_workers=self.workers,
                                pin_memory=self.pin_memory,
                                collate_fn=dataset.collate_fn)
        LOGGER.info('Initialize Dataloader successfully')
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

        Return loss_instance
        """
        LOGGER.info('Initializing LossDetect...')
        loss_instance = loss_class(self.model, self.hyp)
        LOGGER.info('Initialize LossDetect successfully')
        return loss_instance


class TrainDetectMixin(object):
    def __init__(self):
        self.cuda = None
        self.epoch = None
        self.model = None
        self.epochs = None
        self.device = None
        self.scaler = None
        self.loss_fn = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataloader = None

    def train_one_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        loss_name = ('total loss', 'bbox loss', 'class loss', 'object loss')
        loss_all_mean = torch.zeros(4, device=self.device)

        with tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader),
                  bar_format='{l_bar}{bar:10}{r_bar}') as pbar:
            for index, (images, labels, _) in pbar:
                images = images.to(self.device).float() / 255  # to float32 and normalized 0.0-1.0
                labels = labels.to(self.device)

                # TODO warmup in the future

                # forward
                with autocast(enabled=self.cuda):
                    outputs = self.model(images)
                    loss, loss_items = self.loss_fn(outputs, labels, kind='ciou')

                # backward
                self.scaler.scale(loss).backward()

                # optimize
                # maybe accumulating gradient will be better (if index...: ...)
                self.scaler.step(self.optimizer)  # optimizer.step()
                self.scaler.update()
                self.optimizer.zero_grad()

                # mean total loss and loss items
                loss_all = torch.cat((loss, loss_items), dim=0).detach()
                loss_all_mean = self.loss_mean(index, loss_all_mean, loss_all)
                self._show_loss_in_pbar_training(loss_all_mean, pbar)

            # lr_scheduler
            self.lr_scheduler.step()
        return loss_all_mean.tolist(), loss_name

    def _show_loss_in_pbar_training(self, loss, pbar):
        # GPU memory used
        # TODO check whether the cuda memory is right to compute below
        memory_cuda = f'GPU: {torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3f}GB'

        # show in pbar
        space = ' ' * 11
        pbar.set_description_str(f'{space}epoch: {self.epoch}/{self.epochs - 1}, {memory_cuda}')
        pbar.set_postfix_str(f'total loss: {loss[0]:.3f}, '
                             f'bbox loss: {loss[1]:.3f}, '
                             f'class loss: {loss[2]:.3f}, '
                             f'object loss: {loss[3]:.3f}')

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


class FreezeLayersMixin(object):
    def freeze_layers(self, layers):
        raise NotImplementedError


class ValDetectMixin(object):
    def __init__(self):
        self.time = None
        self.seen = None
        self.half = None
        self.model = None
        self.device = None
        self.loss_fn = None
        self.dataloader = None

    def val_once(self):
        loss_name = ('total loss', 'bbox loss', 'class loss', 'object loss')
        loss_all_mean = torch.zeros(4, device=self.device).half() if self.half else torch.zeros(4, device=self.device)
        iou_vector = torch.linspace(0.5, 0.95, 10, device=self.device)
        stats = []  # statistics to save tuple(pred_iou_level, pred conf, pred cls, label cls)
        with tqdm(enumerate(self.dataloader), total=len(self.dataloader),
                  bar_format='{l_bar:>42}{bar:10}{r_bar}') as pbar:
            for index, (images, labels, shape_converts) in pbar:
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

                # get metrics data
                _stats = self._get_metrics_stats(predictions, labels, shape_converts, iou_vector)
                stats.extend(_stats)
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
        outputs = parse_outputs_yolov5(outputs, self.model.anchors, self.model.scalings)  # bbox is xyxy
        # TODO get it by predictions or outputs through below whether used outputs over
        # TODO think it in memory for filter_outputs2predictions
        outputs = filter_outputs2predictions(outputs, 0.25)  # list which len is bs
        return outputs

    def _get_metrics_stats(self, predictions, labels, shape_converts, iou_vector):
        # save tuple(pred_iou_level, pred conf, pred cls, label cls)
        stats = []  # save temporarily
        for index, pred in enumerate(predictions):
            self.seen += 1
            label = labels[labels[:, 0] == index, 1:]  # one image label
            nl = label.shape[0]  # number of label
            label_cls = label[:, 0].tolist() if nl else []

            if not pred.shape[0]:
                if nl:
                    pred_iou_level = torch.zeros((0, iou_vector.shape[0]), dtype=torch.bool)
                    stats.append((pred_iou_level, torch.Tensor(), torch.Tensor(), label_cls))
                continue

            pred[:, :4] = rescale_xyxy(pred[:, :4], shape_converts[index])

            if nl:
                label[:, 1:] = xywh2xyxy(label[:, 1:])
                label[:, 1:] = rescale_xyxy(label[:, 1:], shape_converts[index])
                pred_iou_level = match_pred_label_iou_vector(pred, label, iou_vector)
                # TODO confusion matrix needed
            else:
                pred_iou_level = torch.zeros((pred.shape[0], iou_vector.shape[0]), dtype=torch.bool)
            stats.append((pred_iou_level.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), label_cls))
        return stats

    @staticmethod
    def _show_loss_in_pbar_validating(loss, pbar):
        # GPU memory used
        # TODO check whether the cuda memory is right to compute below
        memory_cuda = f'GPU: {torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3f}GB'

        # show in pbar
        space = ' ' * 11
        pbar.set_description_str(f'{space}validating: {memory_cuda}')
        pbar.set_postfix_str(f'total loss: {loss[0]:.3f}, '
                             f'bbox loss: {loss[1]:.3f}, '
                             f'class loss: {loss[2]:.3f}, '
                             f'object loss: {loss[3]:.3f}')


class ResultsDealDetectMixin(object):
    def __init__(self):
        self.epoch = None
        self.results = None
        self.datasets = None
        self.save_dict = None

    @staticmethod
    def get_results_dict(*args: tuple):
        r"""
        Get dict of results for saving.
        Args:
            *args: (name, type to save)

        Return results dict(for self.results)
        """
        results = {}
        for name, value in args:
            results[name] = value
        return results

    def add_results_dict(self, *args: tuple):
        if self.results is None:
            raise AttributeError('Not find self.results, add failed')
        assert isinstance(self.results, dict), f'Except the type of self.results is dict but got {type(self.results)}'
        for name, value in args:
            self.results[name] = value

    def deal_results_memory(self, results_train, results_val):
        train_loss = results_train
        val_loss, metrics, fps_time = results_val
        (ap50_95, ap50, ap75, ap), (mf1, f1), (mp, p), (mr, r), (cls, cls_number) = metrics

        # train_val
        if self.epoch == 0:
            title_train_val = ['train:', *train_loss[1], 'val:', *val_loss[1],
                               'ap50', 'ap75', 'ap50_95', 'mf1', 'mp', 'mr']
            self.results['train_val_results'].append(title_train_val)
        data_train_val = [*train_loss[0], *val_loss[0], ap50, ap75, ap50_95, mf1, mp, mr]
        self.results['train_val_results'].append(data_train_val)

        # all class
        title_all_class = ['cls_name', 'cls_number', '(IoU=0.50:0.95):', 'AP', 'F1', 'P', 'R']
        if ap is not None:
            rest = (cls, ap, f1, p, r)
            self.results['all_class_results'].append(title_all_class)
            for c, ap_c, f1_c, p_c, r_c in zip(*rest):
                name_c = self.datasets['names'][c]
                number_c = cls_number[c]
                data_all_class = [name_c, number_c, ap_c, f1_c, p_c, r_c]
                self.results['all_class_results'].append(data_all_class)
        else:
            nc = cls_number.shape[0]
            self.results['all_class_results'].append(title_all_class)
            for c in range(nc):
                name_c = self.datasets['names'][c]
                number_c = cls_number[c]
                data_all_class = [name_c, number_c, None, None, None, None]
                self.results['all_class_results'].append(data_all_class)

        # return for computing best_fitness
        return (mp[0], mr[0], mf1[0], ap50, ap50_95) if ap50 is not None else (0,) * 5

    def save_all_results(self):
        r"""Save all content in results then empty self.results"""
        if self.results:
            to_save = []
            for k, v in self.results.items():
                to_save.append((v, self.save_dict[k]))
            save_all_txt(*to_save)
            self.results = {}
        else:
            LOGGER.warning('The self.results is empty, save nothing')


if __name__ == '__main__':
    pass
