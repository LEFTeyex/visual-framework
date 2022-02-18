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
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.log import LOGGER, add_log_file
from utils.check import check_only_one_set
from utils.general import delete_list_indices
from utils.typeslib import _str_or_None, _module_or_None, _optimizer, _lr_scheduler, _pkt_or_None, _dataset_c

__all__ = ['SetSavePathMixin', 'LoadAllCheckPointMixin', 'DataLoaderMixin']


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


class LoadAllCheckPointMixin(object):
    r"""
    Need self.model, self.device, self.weights, self.checkpoint, self.param_groups, self.epochs
    The function in the Mixin below.
    1. Load checkpoint from self.weights '*.pt' or '*.pth' file. ---------- ues self.device, self.weights
    2. Load model from 'model' or 'state_dict' or initialize model. ------- ues self.device, self.checkpoint
    3. Load optimizer from state_dict and add param_groups first. --------- ues self.checkpoint, self.param_groups
    4. Load lr_scheduler from state_dict. --------------------------------- ues self.checkpoint
    5. Load start_epoch. -------------------------------------------------- ues self.checkpoint, self.epochs
    6. Load best_fitness for choosing which weights of model is best among epochs. --- ues self.checkpoint
    7. Set param_groups, need to know how to set param_kind_list. --------- ues self.model
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
                          'lr_scheduler_state_dict':  scheduler.state_dict(),
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

    def load_lr_scheduler(self, scheduler_instance: _lr_scheduler, load: bool = True):
        r"""
        Load lr_scheduler from state_dict.
        Args:
            scheduler_instance: _lr_scheduler = StepLR(self.optimizer, 30) etc. the instance of lr_scheduler
            load: bool = True / False, Default=True(load lr_scheduler state_dict)

        Return scheduler_instance
        """
        LOGGER.info('Initialize lr_scheduler successfully')
        if load:
            # load lr_scheduler
            LOGGER.info('Loading lr_scheduler state_dict...')
            self._check_checkpoint_not_none()
            scheduler_instance.load_state_dict(self.checkpoint['lr_scheduler_state_dict'])
            LOGGER.info('Load lr_scheduler state_dict successfully...')

        else:
            LOGGER.info('Do not load lr_scheduler state_dict')
        return scheduler_instance

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

        Return param_groups
        """
        # do not set param_groups
        if param_kind_tuple is None:
            LOGGER.info('No param_groups set')
            return None
        # set param_groups
        LOGGER.info('Setting param_groups...')
        # TODO: Upgrade the algorithm in the future for filtering parameters from model.modules() better
        param_groups = []
        rest = []  # save the rest parameters that are not filtered
        indices = []  # save parameters (index, name) temporarily to delete

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
            save = []  # save parameters temporarily

            # filter parameters from model
            for index, module in enumerate(rest):
                if hasattr(module, name):  # has module.weight / module.bias
                    module_name = eval(str_eval)  # module.weight / module.bias

                    if isinstance(module, kind):
                        indices[index] = indices[index].replace(name, '')
                        save.append(module_name)

                    elif isinstance(module_name, kind):
                        indices[index] = indices[index].replace(name, '')
                        save.append(module_name)

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
            param_dict['params'] = save
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


class DataLoaderMixin(object):
    r"""
    Need self.datasets, self.image_size, self.batch_size, self.shuffle, self.workers, self.pin_memory
    The function in the Mixin below.
    1. Set dataset and get dataloader. --- all self.*
    """

    def __init__(self):
        self.datasets = None
        self.image_size = None
        self.batch_size = None
        self.shuffle = None
        self.workers = None
        self.pin_memory = None

    def get_dataloader(self, dataset: _dataset_c, name: str):
        r"""
        Set dataset and get dataloader.
        Args:
            dataset: _dataset = dataset class
            name: str = 'train' / 'val' / 'test'

        Return dataloader
        """
        LOGGER.info('Initializing Dataloader...')
        # set dataset
        dataset = dataset(self.datasets[name], self.image_size, name)

        # set dataloader
        # TODO upgrade num_workers(deal and check) and sampler(distributed.DistributedSampler) for DDP
        batch_size = min(self.batch_size, len(dataset))
        dataloader = DataLoader(dataset, batch_size, self.shuffle,
                                num_workers=self.workers,
                                pin_memory=self.pin_memory,
                                collate_fn=dataset.collate_fn)
        LOGGER.info('Initialize Dataloader successfully')
        return dataloader


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


if __name__ == '__main__':
    pass
