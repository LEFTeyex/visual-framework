r"""
Mixin Module.
Consist of all Mixin class.
You can use the mixin module building train model(class), it is flexible, easy and quick.
The built example in train*.py.

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
from datetime import datetime
from pycocotools.coco import COCO
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval
from torch.utils.tensorboard import SummaryWriter

from . import WRITER
from .log import LOGGER, log_fps_time
from .swa import update_bn
from .metrics import compute_fps
from .check import check_only_one_set
from .general import delete_list_indices, time_sync, loss_to_mean, HiddenPrints, hasattr_not_none
from .typeslib import strpath, instance_, module_, dataset_, dataloader_, module_or_None, \
    str_or_None, tuple_or_list, pkt_or_None

__all__ = [
    'SetSavePathMixin',
    'TensorboardWriterMixin',
    'SaveCheckPointMixin',
    'LoadAllCheckPointMixin',
    'FreezeLayersMixin',
    'DataLoaderMixin',
    'TrainMixin',
    'ValMixin',
    'COCOEvaluateMixin',
    'ReleaseMixin'
]


class SetSavePathMixin(object):
    r"""
    Methods:
        1. set_save_path --- need all self.*.
    """

    def __init__(self):
        self.save_name = None
        self.save_path = None

    def set_save_path(self, *args: tuple_or_list) -> dict:
        r"""
        Set save path to path_dict.
        Args:
            args: tuple_or_list = (save_name, path), ...

        Returns:
            dict{'name': strpath, ...}
        """
        LOGGER.info('Setting save path...')
        self._set_save_name()
        self.save_path /= self.save_name
        self.save_path.mkdir(parents=True, exist_ok=True)

        # set save (name, path) to dict
        path_dict = dict()
        for k, v in args:
            path = self.save_path / v
            path.parent.mkdir(parents=True, exist_ok=True)
            path_dict[k] = str(path)
        LOGGER.info('Set save path successfully')
        return path_dict

    def _set_save_name(self):
        r"""Set save_name for save file without repeating (for examole: runs/*/exp1, runs/*/exp2)"""
        self.save_path = Path(self.save_path)
        if self.save_path.exists():
            # to save number of exp
            list_num = []
            for dir_name in self.save_path.glob(f'{self.save_name}*'):  # get all dir matching (exp*)
                if dir_name.is_dir():
                    dir_name = str(dir_name.name).lstrip(self.save_name)
                    try:
                        list_num.append(int(dir_name))
                    except ValueError:
                        # no really error
                        LOGGER.error(f'The {dir_name} is an unexpected name '
                                     f'in end of directory that starts with exp')
            if list_num:
                self.save_name += str(max(list_num) + 1)  # set save_name, for example exp6
            else:
                self.save_name += '1'  # set save_name exp1

        else:
            self.save_name += '1'  # set save_name exp1


class TensorboardWriterMixin(object):
    r"""
    Methods:
        1. set_tensorboard_writer --- need self.tensorboard.
        2. close_tensorboard --- need self.writer.
    """

    def __init__(self):
        self.writer = None
        self.tensorboard = None

    def set_tensorboard_writer(self, path: strpath):
        r"""
        Set tensorboard for self.writer.
        Args:
            path: strpath = StrPath to save tensorboard file.
        """
        if self.tensorboard:
            LOGGER.info('Setting tensorboard writer...')
            writer = SummaryWriter(str(path))
            LOGGER.info('Setting tensorboard successfully')
            LOGGER.info(f"See tensorboard results please run "
                        f"'tensorboard --logdir={path}' in Terminal")
        else:
            writer = None
            LOGGER.info('Set tensorboard writer none')
        return writer

    def close_tensorboard(self):
        r"""Close writer which is the instance of SummaryWriter in tensorboard"""
        if self.writer:
            self.writer.flush()
            self.writer.close()


class SaveCheckPointMixin(object):
    r"""
    Methods:
        1. get_checkpoint --- need self.* except self.epochs.
        2. save_checkpoint_best_last --- need all self.*.
        3. save_checkpoint --- need self.* except self.epochs.
    """

    def __init__(self):
        self.model = None
        self.epoch = None
        self.epochs = None
        self.scaler = None
        self.optimizer = None
        self.swa_model = None
        self.lr_scheduler = None
        self.best_fitness = None
        self.warmup_lr_scheduler = None

    def get_checkpoint(self, model_state_dict: bool = False) -> dict:
        # de_parallel model in the future
        r"""
        Get checkpoint to save.
        Need different checkpoint to save please override the get_checkpoint method.
        Args:
            model_state_dict: bool = if True, checkpoint will save model.state_dict().

        Returns:
            checkpoint
        """
        LOGGER.debug('Getting checkpoint...')

        checkpoint = {
            'model': self.model.float().state_dict() if model_state_dict else self.model.float(),
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }

        if hasattr_not_none(self, 'swa_model'):
            checkpoint['swa_model'] = self.swa_model.module.float().state_dict() if model_state_dict \
                else self.swa_model.module.float()
            checkpoint['n_averaged'] = self.swa_model.n_averaged

        if hasattr_not_none(self, 'scaler'):
            checkpoint['scaler'] = self.scaler.state_dict()

        if hasattr_not_none(self, 'warmup_lr_scheduler'):
            checkpoint['warmup_lr_scheduler'] = self.warmup_lr_scheduler.state_dict()

        LOGGER.debug('Get checkpoint successfully')
        return checkpoint

    def save_checkpoint_best_last(self, fitness, best_path: strpath, last_path: strpath, model_state_dict: bool):
        r"""
        Save checkpoint when get better fitness or at last.
        Override get_checkpoint if necessary and the save checkpoint will be changed.
        Args:
            fitness: = a number to compare with best_fitness.
            best_path: strpath = StrPath to save best.pt.
            last_path: strpath = StrPath to save last.pt.
            model_state_dict: bool = if True, checkpoint will save model.state_dict().
        """
        # save the best checkpoint
        if fitness > self.best_fitness:
            LOGGER.debug(f'Saving best checkpoint in epoch{self.epoch}...')
            self.best_fitness = fitness
            torch.save(self.get_checkpoint(model_state_dict), best_path)
            LOGGER.debug(f'Save best checkpoint in epoch{self.epoch} successfully')

        # save the last checkpoint
        if self.epoch + 1 == self.epochs:
            LOGGER.info('Saving last checkpoint')
            torch.save(self.get_checkpoint(model_state_dict), last_path)
            LOGGER.info('Save last checkpoint successfully')

    def save_checkpoint(self, save_path: strpath, model_state_dict: bool):
        r"""Save checkpoint straightly"""
        LOGGER.info('Saving checkpoint')
        torch.save(self.get_checkpoint(model_state_dict), save_path)
        LOGGER.info('Save checkpoint successfully')


class LoadAllCheckPointMixin(object):
    r"""
    Methods:
        1. load_checkpoint --- need self.device.
        2. load_model --- need self.device, self.checkpoint.
        3. load_swa_model --- need self.device, self.checkpoint.
        4. load_state_dict --- need self.checkpoint.
        5. load_start_epoch --- need self.checkpoint, self.epochs.
        6. load_best_fitness --- need self.checkpoint.
        7. set_param_groups --- need self.model.
    """

    def __init__(self):
        self.model = None
        self.device = None
        self.epochs = None
        self.checkpoint = None

    def load_checkpoint(self, path: strpath, suffix: tuple = ('.pt', '.pth')) -> dict or None:
        r"""
        Load checkpoint from path '*.pt' or '*.pth' file.

        Args:
            path: strpath = the strpath of checkpoint.
            suffix: tuple = ('.pt', '.pth', ...) the suffix of checkpoint file.

        Returns:
            checkpoint or None
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
            return

    def load_model(self,
                   model_instance: module_or_None = None,
                   load_key: str_or_None = None,
                   state_dict_operation: bool = True,
                   load: str_or_None = 'state_dict') -> module_:
        r"""
        Load model from 'model' or 'state_dict' or initialize model.
        Args:
            model_instance: module_or_None = the instance of model.
            load_key: str_or_None = the key of model in self.checkpoint.
            state_dict_operation: bool = if True, do .state_dict() operation.
            load: str_or_None = the load way of model, consist of None / 'model' / 'state_dict'.

        Returns:
            model_instance
        """
        # check whether model_instance, state_dict_operation and load are conflict
        if load == 'model':
            if not check_only_one_set(model_instance, load):
                raise ValueError(f'Only one of {model_instance} '
                                 f'and {load} can be set, please reset them')
        elif model_instance is None:
            raise ValueError("The model_instance can not be None,"
                             " if load = None or 'state_dict'")

        if load == 'model' and state_dict_operation:
            state_dict_operation = False
            LOGGER.info(f'Correct state_dict_operation=False, '
                        f'please notice the load: {load} and '
                        f'state_dict_operation: {state_dict_operation} is conflict now')

        # get checkpoint to load
        to_load = None
        if load is not None and load_key is not None:
            self._check_checkpoint_not_none()
            to_load = self.checkpoint[load_key].state_dict() if state_dict_operation else self.checkpoint[load_key]

        if load is None:
            LOGGER.info(f'Initialize {load_key} successfully')
            # model need to initial itself in its __init__()
            LOGGER.info(f'Load nothing to {load_key}')

        elif load == 'state_dict':
            # delete the same keys but different weight shape
            model_ns = ((name, weight.shape) for name, weight in model_instance.state_dict().items())
            for name, shape in model_ns:
                if name in to_load and to_load[name].shape != shape:
                    del to_load[name]
                    LOGGER.warning(f'Delete the {name} in state_dict because of different weight shape')

            # load model and get the rest of (missing_keys, unexpected_keys)
            model_instance = self._load_state_dict(model_instance, to_load, name_log=type(model_instance).__name__)

        elif load == 'model':
            LOGGER.info(f'Loading total {load_key} from self.checkpoint...')
            model_instance = to_load
            LOGGER.info(f'Load total {load_key} successfully')

        else:
            raise ValueError(f"The arg load: {load} do not match, "
                             f"please input one of  (None, 'model', 'state_dict')")
        return model_instance.to(self.device).float()  # return float model to self.device

    def load_swa_model(self,
                       model_instance: module_or_None = None,
                       load_key: str_or_None = None,
                       state_dict_operation: bool = True,
                       load: str_or_None = 'state_dict',
                       load_n_averaged_key: str_or_None = None) -> module_:
        r"""
        Load model from 'model' or 'state_dict' or initialize model.
        Args:
            model_instance: module_or_None = the instance of model.
            load_key: str_or_None = the key of model in self.checkpoint.
            state_dict_operation: bool = if True, do .state_dict() operation.
            load: str_or_None = the load way of model, consist of None / 'model' / 'state_dict'.
            load_n_averaged_key: str_or_None = the key of n_averaged in checkpoint if load it.

        Returns:
            model_instance
        """
        model_instance.module = self.load_model(model_instance.module, load_key, state_dict_operation, load)

        if load_n_averaged_key:
            LOGGER.info(f'Loading {load_key} n_averaged...')
            self._check_checkpoint_not_none()
            model_instance.n_averaged = self.checkpoint[load_n_averaged_key]
            LOGGER.info(f'Load {load_key} n_averaged successfully')
        return model_instance

    def load_state_dict(self, instance: instance_, load_key: str_or_None = None) -> instance_:
        r"""
        Load state_dict for instance.
        Args:
            instance: instance_ = instance to load_state_dict.
            load_key: str_or_None = the key in self.checkpoint to load, load nothing if None.

        Returns:
            instance
        """
        name = type(instance).__name__
        if load_key:
            self._check_checkpoint_not_none()
            instance = self._load_state_dict(instance, self.checkpoint[load_key], name_log=name)

        else:
            LOGGER.info(f'Load nothing for {name} state_dict')

        return instance

    def load_start_epoch(self, load_key: str_or_None = None, load: str_or_None = 'continue') -> int:
        r"""
        Load start_epoch.
        Args:
            load_key: str_or_None = the key in self.checkpoint to load, initialize start_epoch if None.
            load: str_or_None = the load way, consist of 'continue' / 'add' / None, Default='continue'.
                continue: continue to train from start_epoch to self.epochs.
                add: train self.epochs more times.
                None: initialize start_epoch=0.

        Returns:
            start_epoch
        """
        if load is not None and load_key is None:
            raise ValueError(f'The load_key can not be None when the load={load}')

        to_load = None
        if load_key is not None:
            self._check_checkpoint_not_none()
            to_load = self.checkpoint[load_key]

        if load is None:
            # initialize start_epoch
            LOGGER.info('Initializing start_epoch...')
            start_epoch = 0
            LOGGER.info(f'Initialize start_epoch={start_epoch} successfully')
            LOGGER.info(f'The Model will be trained {self.epochs} epochs')

        elif load == 'continue':
            # load start_epoch to continue
            LOGGER.info('Loading start_epoch to continue...')

            start_epoch = to_load + 1
            if self.epochs < start_epoch:
                raise ValueError(f'The epochs: {self.epochs} can not be '
                                 f'less than the start_epoch: {start_epoch}')
            LOGGER.info(f'Load start_epoch={start_epoch} to continue successfully')
            LOGGER.info(f'The Model will be trained {self.epochs - start_epoch + 1} epochs')

        elif load == 'add':
            # load start_epoch to add epochs to train
            LOGGER.info('Loading start_epoch to add epochs to train...')
            start_epoch = to_load + 1
            self.epochs += to_load
            LOGGER.info(f'Load start_epoch={start_epoch} to add epochs to train successfully')
            LOGGER.info(f'The Model will be trained {self.epochs} epochs')

        else:
            raise ValueError(f"The arg load: {load} do not match, "
                             f"please input one of  (None, 'continue', 'add')")
        return start_epoch

    def load_best_fitness(self, load_key: str_or_None = None) -> float:
        r"""
        Load best_fitness for choosing which weights of model is best among epochs.
        Args:
            load_key: str_or_None = the key in self.checkpoint to load, initialize best_fitness if None.

        Returns:
            best_fitness
        """
        if load_key:
            # load best_fitness
            LOGGER.info('Loading best_fitness...')
            self._check_checkpoint_not_none()
            best_fitness = self.checkpoint[load_key]
            LOGGER.info('Load best_fitness successfully')

        else:
            # initialize best_fitness
            LOGGER.info('Initializing best_fitness...')
            best_fitness = 0.0
            LOGGER.info(f'Initialize best_fitness={best_fitness} successfully')
        return best_fitness

    def set_param_groups(self, param_kind_tuple: pkt_or_None = None) -> list or None:
        # Upgrade the algorithm in the future for filtering parameters from model.modules() better
        r"""
        Set param_groups, need to know how to set param_kind_list.
        The parameters in param_groups will not repeat.
        Args:
            param_kind_tuple: pkt_or_None = (
                                              ('bias', nn.Parameter, {lr=0.01}),
                                              ('weight', nn.BatchNorm2d, {lr=0.02})
                                            ).

        Returns:
            param_groups [{'params': nn.Parameter, 'lr': 0.01, ...}, ...]
        """
        # do not set param_groups
        if param_kind_tuple is None:
            LOGGER.info('Set param_groups None')
            return
        # set param_groups
        LOGGER.info('Setting param_groups...')
        param_groups = []
        rest = []  # save_params the rest parameters that are not filtered
        indices = []  # save_params parameters (name, ...) temporarily to delete

        # get all parameters from model and set one of ('weightbias', 'weight', 'bias') to its indices
        for element in self.model.modules():
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
            str_eval = f'module.{name}'  # for eval to get code module.weight or module.bias
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

            # to record in tensorboard for lr corresponding to params
            param_dict['name'] = f'{kind.__name__}.{name}'

            param_groups.append(param_dict)

        # check whether rest is empty
        if rest:
            LOGGER.exception(f'There are still {len(rest)} modules ungrouped,'
                             f' please reset the arg param_kind_tuple')
            for module, param_kind in (rest, indices):
                LOGGER.info(f'{module} --- {param_kind}')

        LOGGER.info('Set param_groups successfully')
        return param_groups

    @staticmethod
    def _load_state_dict(instance: instance_, state_dict, name_log: str, strict: bool = False) -> instance_:
        r"""Load state_dict in detail for logger"""
        LOGGER.info(f'Loading {name_log} state_dict...')
        rest = instance.load_state_dict(state_dict, strict=strict)
        missing_keys, unexpected_keys = rest
        if missing_keys or unexpected_keys:
            LOGGER.warning(f'There are the rest of {len(missing_keys)} missing_keys'
                           f' and {len(unexpected_keys)} unexpected_keys when load {name_log}')
            LOGGER.warning(f'missing_keys: {missing_keys}')
            LOGGER.warning(f'unexpected_keys: {unexpected_keys}')
            LOGGER.info(f'Load {name_log} state_dict successfully with the rest of keys')
        else:
            LOGGER.info(f'Load {name_log} state_dict successfully without the rest of keys')
        return instance

    def _check_checkpoint_not_none(self):
        r"""Check whether self.checkpoint exists"""
        if self.checkpoint is None:
            raise ValueError('The self.checkpoint is None, please load checkpoint')


class FreezeLayersMixin(object):
    r"""
    Methods:
        1. freeze_layers --- need self.model.
        2. unfreeze_layers --- need self.model.
        3. freeze_model --- need self.model.
        4. unfreeze_model --- need self.model.
    """

    def __init__(self):
        self.model = None

    def freeze_layers(self, layer_names: list):
        r"""
        Freeze layers in model by names.
        Args:
            layer_names: list = list consist of name in model layers.
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
            LOGGER.info('Frozen layers successfully')

    def unfreeze_layers(self, layer_names: list):
        r"""
        Unfreeze layers in model by names.
        Args:
            layer_names: list = list consist of name in model layers.
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
            LOGGER.info('Unfrozen layers successfully')

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
    # upgrade num_workers(deal and check) and sampler(distributed.DistributedSampler) for DDP
    # upgrade add more augments for Dataloader
    r"""
    Methods:
        1. set_dataloader --- need all self.*.
    """

    def __init__(self):
        self.workers = None
        self.batch_size = None
        self.pin_memory = None
        self.visual_image = None

    def set_dataloader(self, dataset_instance: dataset_, shuffle: bool = False) -> dataloader_:
        r"""
        Set dataloader.
        Args:
            dataset_instance: dataset_ = dataset instance.
            shuffle: bool = False/True.

        Returns:
            dataloader instance
        """
        LOGGER.info(f'Initialize Dataloader...')

        # visualizing
        if self.visual_image:
            if hasattr(dataset_instance, 'name'):
                name = dataset_instance.name
            else:
                name = 'dataset'
            self.visual_dataset(dataset_instance, name)

        # set dataloader
        if hasattr(dataset_instance, 'collate_fn'):
            collate_fn = dataset_instance.collate_fn
        else:
            collate_fn = None
        batch_size = min(self.batch_size, len(dataset_instance))

        dataloader = DataLoader(dataset_instance,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=self.workers,
                                pin_memory=self.pin_memory,
                                collate_fn=collate_fn)
        LOGGER.info(f'Initialize Dataloader successfully')
        return dataloader

    def visual_dataset(self, dataset_instance, name: str):
        r"""Set visual way for dataset"""
        # WRITER.add_datasets_images_labels_detect(self.writer, dataset, name) for detection
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


class _TrainValMixin(object):
    r"""Consist of universal methods for training or validating iteration"""

    def __init__(self):
        self.model = None
        self.loss_fn = None

    def forward_in_model(self, x, data_dict):
        r"""Forward in model and has an interface data_dict"""
        x = self.model(x)
        return x, data_dict

    def compute_loss(self, outputs, labels, data_dict):
        r"""Compute loss for backward and has an interface data_dict"""
        loss, *other_loss = self.loss_fn(outputs, labels)
        return loss, other_loss, data_dict

    @staticmethod
    def mean_loss(index, loss_mean, loss, data_dict):
        r"""Compute mean loss to show and has an interface data_dict"""
        loss = loss.detach()
        loss_mean = loss_to_mean(index, loss_mean, loss)
        return loss_mean, data_dict


class TrainMixin(_TrainValMixin):
    r"""
    Methods:
        1. train_one_epoch --- need all self.*.
        2. other methods can be overridden to change the operation mode.
    """

    def __init__(self):
        super(TrainMixin, self).__init__()
        self.inc = None
        self.cuda = None
        self.swa_c = None
        self.model = None
        self.epoch = None
        self.writer = None
        self.epochs = None
        self.scaler = None
        self.device = None
        self.loss_fn = None
        self.optimizer = None
        self.swa_model = None
        self.image_size = None
        self.visual_graph = None
        self.lr_scheduler = None
        self.swa_scheduler = None
        self.swa_start_epoch = None
        self.train_dataloader = None
        self.warmup_lr_scheduler = None

    def train_one_epoch(self, loss_name: tuple_or_list) -> dict:
        r"""
        Finish train one epoch.
        It is flexible to change the operation mode by overriding the method.
        The data_dict is the interface for extra data or other operation.
        Args:
            loss_name: tuple_or_list = the name of loss corresponding to the loss to show.

        Returns:
            data_dict
        """
        # data_dict to save the data, and it is also an interface
        # data_dict consist of (index, loss_name, loss_mean, other_data_iter, other_loss_iter)
        data_dict = {
            'index': None,
            'loss_name': loss_name,
            'loss_mean': torch.tensor(0., device=self.device),
            'other_data_iter': None,
            'other_loss_iter': None,
        }
        data_dict = self.preprocess(data_dict)  # preprocess for data_dict
        with tqdm(enumerate(self.train_dataloader),
                  total=len(self.train_dataloader),
                  bar_format=self.bar_format_train()) as pbar:
            for index, data in pbar:
                data_dict['index'] = index

                # preprocess in iteration
                x, labels, data_dict['other_data_iter'], data_dict = self.preprocess_iter(data, data_dict)

                # forward with mixed precision
                with autocast(enabled=self.cuda):
                    outputs, data_dict = self.forward_in_model(x, data_dict)
                    loss, data_dict['other_loss_iter'], data_dict = self.compute_loss(outputs, labels, data_dict)

                # backward and optimize
                self.backward_optimize(loss)

                # warmup_lr_scheduler step
                self.warmup_lr_scheduler_step()

                # mean loss
                data_dict['loss_mean'], data_dict = self.mean_loss(index, data_dict['loss_mean'], loss, data_dict)
                data_dict = self.show_in_pbar(pbar, data_dict)

                # postprocess in iteration for data_dict
                data_dict = self.postprocess_iter(data_dict)

        # lr_scheduler step without warmup_lr_scheduler
        self.lr_scheduler_step_without_warmup()

        # upgrade swa_model
        self.swa_model_update()

        # postprocess for data_dict
        data_dict = self.postprocess(data_dict)
        return data_dict

    @staticmethod
    def bar_format_train():
        r"""The bar_format in tqdm"""
        return '{l_bar}{bar:10}{r_bar}'

    def preprocess(self, data_dict) -> dict:
        r"""It is a preprocessing in the epoch and has an interface data_dict"""
        self.model.train()
        self.optimizer.zero_grad()
        return data_dict

    def preprocess_iter(self, data, data_dict):
        r"""It is a preprocessing in training iteration and has an interface data_dict"""
        x, labels, *other_data = data
        x = x.to(self.device).float()
        labels = labels.to(self.device)

        # visual in tensorboard pytorch
        if hasattr_not_none(self, 'writer') and self.epoch == (self.epoch + data_dict['index']):
            WRITER.add_optimizer_lr(self.writer, self.optimizer, self.epoch)
        return x, labels, other_data, data_dict

    def postprocess_iter(self, data_dict) -> dict:
        r"""It is a postprocessing in training iteration and has an interface data_dict"""
        return data_dict

    def postprocess(self, data_dict) -> dict:
        r"""It is a postprocessing in the epoch and has an interface data_dict"""
        # visual in tensorboard pytorch
        if hasattr_not_none(self, 'writer'):
            loss_mean, loss_name = data_dict['loss_mean'], data_dict['loss_name']
            WRITER.add_epoch_curve(self.writer, 'train_loss', loss_mean, loss_name, self.epoch)

            if self.visual_graph:
                WRITER.add_model_graph(self.writer, self.model, self.inc, self.image_size, self.epoch)
        return data_dict

    def backward_optimize(self, loss):
        # maybe accumulating gradient will be better (if index...)
        r"""Backward loss and optimize parameters in model"""
        if hasattr_not_none(self, 'scaler'):
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)  # optimizer.step()
            self.scaler.update()
            self.optimizer.zero_grad()  # improve a little performance when set_to_none=True
        else:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def warmup_lr_scheduler_step(self):
        r"""Step warmup lr scheduler if it has the attribute warmup_lr_scheduler"""
        if hasattr_not_none(self, 'warmup_lr_scheduler'):
            if not (hasattr_not_none(self, 'swa_scheduler') and self.swa_start):
                self.warmup_lr_scheduler.step()

    def lr_scheduler_step_without_warmup(self):
        r"""Step lr scheduler if it has no attribute warmup_lr_scheduler"""
        if hasattr_not_none(self, 'lr_scheduler') and not hasattr_not_none(self, 'warmup_lr_scheduler'):
            if not (hasattr_not_none(self, 'swa_scheduler') and self.swa_start):
                self.lr_scheduler.step()

    def swa_model_update(self):
        r"""Update swa_model, bn, attr and step swa_scheduler if it has the attribute swa_model and swa_scheduler"""
        c = round(max(1, self.swa_c if hasattr_not_none(self, 'swa_c') else 1))
        if hasattr_not_none(self, 'swa_model') and self.swa_start and \
                (self.epoch - self.swa_start_epoch) % c == 0:
            self.swa_model.update_parameters(self.model)
            self.update_swa_bn()
            self.update_swa_attr()

        if hasattr_not_none(self, 'swa_scheduler') and self.swa_start:
            self.swa_scheduler.step()

    def update_swa_bn(self):
        r"""Update swa model bn"""
        update_bn(self.train_dataloader, self.swa_model, self.device)

    def update_swa_attr(self):
        r"""
        Update swa_model attr, an example as follow:
        update_attr(self.swa_model.module, self.model, include=(...)).
        """
        if hasattr_not_none(self, 'swa_model'):
            raise NotImplementedError

    @property
    def swa_start(self) -> bool:
        r"""Whether swa model start"""
        return self.epoch >= self.swa_start_epoch

    def show_in_pbar(self, pbar, data_dict) -> dict:
        r"""Show something in pbar and has an interface data_dict"""
        loss_mean, loss_name = data_dict['loss_mean'], data_dict['loss_name']
        # GPU memory used which an accurate value because of 1024 * 1024 * 1024 = 1073741824
        memory = torch.cuda.memory_reserved(self.device) / 1073741824 if torch.cuda.is_available() else 0
        memory_cuda = f'GPU: {memory:.3f}GB'

        # show in pbar
        space = ' ' * 11
        progress = f'{self.epoch}/{self.epochs - 1}:'
        pbar.set_description_str(f"{space}epoch {progress:<9}{memory_cuda}")
        show = ''.join([f'{x}: {y:.5f} ' for x, y in zip(loss_name, loss_mean)])
        pbar.set_postfix_str(show)
        return data_dict


class ValMixin(_TrainValMixin):
    r"""
    Methods:
        1. val_once --- need all self.*.
        2. other methods can be overridden to change the operation mode.
    """

    def __init__(self):
        super(ValMixin, self).__init__()
        self.half = None
        self.model = None
        self.epoch = None
        self.writer = None
        self.device = None
        self.loss_fn = None
        self.dataloader = None

    def val_once(self, loss_name: tuple_or_list) -> dict:
        r"""
        Finish val once.
        It is flexible to change the operation mode by overriding the method.
        The data_dict is the interface for extra data or other operation.
        Args:
            loss_name: tuple_or_list = the name of loss corresponding to the loss to show.

        Returns:
            data_dict
        """
        # data_dict to save the data, and it is also an interface
        # data_dict consist of (index, time, seen, loss_name, loss_mean, stats, other_data_iter, other_loss_iter)
        data_dict = {
            'index': None,
            'time': 0.0,
            'seen': 0,
            'loss_name': loss_name,
            'loss_mean': torch.tensor(0., device=self.device),
            # 'stats_name': [],  # for saving stats data to evaluate in method process_eval_stats
            'other_data_iter': None,
            'other_loss_iter': None,
        }
        data_dict = self.preprocess(data_dict)  # preprocess for data_dict

        with tqdm(enumerate(self.dataloader),
                  total=len(self.dataloader),
                  bar_format=self.bar_format_val()) as pbar:
            for index, data in pbar:
                time_start = time_sync()
                data_dict['index'] = index

                # preprocess in iteration
                x, labels, data_dict['other_data_iter'], data_dict = self.preprocess_iter(data, data_dict)

                # inference
                outputs, data_dict = self.forward_in_model(x, data_dict)

                # compute loss
                loss, data_dict['other_loss_iter'], data_dict = self.compute_loss(outputs, labels, data_dict)

                # mean loss
                data_dict['loss_mean'], data_dict = self.mean_loss(index, data_dict['loss_mean'], loss, data_dict)
                data_dict = self.show_in_pbar(pbar, data_dict)

                # parse outputs
                predictions, data_dict = self.decode_iter(outputs, data_dict)

                # process for evaluating stats and save to data_dict['stats']
                data_dict = self.process_stats(predictions, data_dict)

                # get time difference and seen
                data_dict['time'] += time_sync() - time_start
                data_dict['seen'] += x.shape[0]

                # postprocess in iteration for data_dict
                data_dict = self.postprocess_iter(data_dict)

        # postprocess for data_dict
        data_dict = self.postprocess(data_dict)
        return data_dict

    @staticmethod
    def bar_format_val():
        r"""The bar_format in tqdm"""
        return '{l_bar:>42}{bar:10}{r_bar}'

    def preprocess(self, data_dict) -> dict:
        r"""It is a preprocessing in the validating and has an interface data_dict"""
        self.model.eval()
        return data_dict

    def preprocess_iter(self, data, data_dict):
        r"""It is a preprocessing in validating iteration and has an interface data_dict"""
        x, labels, *other_data = data
        x = x.to(self.device)
        labels = labels.to(self.device)
        x = x.half() if self.half else x.float()
        return x, labels, other_data, data_dict

    def postprocess_iter(self, data_dict) -> dict:
        r"""It is a postprocessing in validating iteration and has an interface data_dict"""
        return data_dict

    def postprocess(self, data_dict) -> dict:
        r"""It is a postprocessing in the validating and has an interface data_dict"""
        # visual in tensorboard pytorch
        if hasattr_not_none(self, 'writer'):
            loss_mean, loss_name = data_dict['loss_mean'], data_dict['loss_name']
            WRITER.add_epoch_curve(self.writer, 'val_loss', loss_mean, loss_name, self.epoch)

        # compute fps time and make it log
        fps_time = compute_fps(data_dict['seen'], data_dict['time'])
        log_fps_time(fps_time)
        return data_dict

    def decode_iter(self, outputs, data_dict):
        r"""Decode outputs from model and has an interface data_dict"""
        return outputs, data_dict

    def process_stats(self, predictions, data_dict) -> dict:
        r"""The interface of processing stats for evaluating"""
        return data_dict

    def show_in_pbar(self, pbar, data_dict) -> dict:
        r"""Show something in pbar and has an interface data_dict"""
        loss_mean, loss_name = data_dict['loss_mean'], data_dict['loss_name']
        memory = torch.cuda.memory_reserved(self.device) / 1073741824 if torch.cuda.is_available() else 0
        memory_cuda = f'GPU: {memory:.3f}GB'

        # show in pbar
        space = ' ' * 11
        pbar.set_description_str(f"{space}{'validating:':<15}{memory_cuda}")
        show = ''.join([f'{x}: {y:.5f} ' for x, y in zip(loss_name, loss_mean)])
        pbar.set_postfix_str(show)
        return data_dict


class COCOEvaluateMixin(object):
    r"""
    Methods:
        1. coco_evaluate
        2. save_coco_results
        3. append_json_dt
        4. empty_append
    """

    @staticmethod
    def coco_evaluate(coco_gt: strpath,
                      coco_dt,
                      img_ids: list,
                      eval_type: str = 'bbox',
                      print_result: bool = False):
        r"""
        Evaluate by coco.
        Args:
            coco_gt: strpath = StrPath of coco_gt json.
            coco_dt: = StrPath of coco_dt json / coco_dt list.
            img_ids: list = image id to evaluate.
            eval_type: str = evaluate type consist of ('segm', 'bbox', 'keypoints').
            print_result: bool = whether print result in COCO.

        Returns:
            (coco.eval, coco.stats)
        """

        if not Path(coco_gt).exists():
            raise FileExistsError(f'The coco_gt {coco_gt} json file do not exist')

        if not isinstance(coco_dt, list) and not Path(coco_gt).exists():
            raise FileExistsError(f'The coco_dt {coco_dt} json file do not exist')

        with HiddenPrints(print_result):
            coco_gt = COCO(coco_gt)
            coco_dt = coco_gt.loadRes(coco_dt)
            coco = COCOeval(coco_gt, coco_dt, iouType=eval_type)
            coco.params.imgIds = img_ids
            coco.evaluate()
            coco.accumulate()
            coco.summarize()
        return coco.eval, coco.stats

    @staticmethod
    def save_coco_results(coco_eval, path: strpath):
        r"""
        Save coco_results.
        Args:
            coco_eval: = coco.eval.
            path: strpath = save path.
        """
        path = str(path)
        params = coco_eval['params']
        new_params = {}
        for k, v in params.__dict__.items():
            if isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, (list, tuple)):
                v = np.asarray(v).tolist()
            new_params[k] = v
        coco_eval['params'] = new_params

        for k, v in coco_eval.items():
            if isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, datetime):
                v = str(v)
            coco_eval[k] = v

        with open(path, 'w') as f:
            json.dump(coco_eval, f)
            LOGGER.info(f"Save coco results json {path} successfully")

    @staticmethod
    def append_json_dt(bboxes, scores, cls_ids, image_id: int, json_dt: list):
        r"""
        Append predictions to json_dt list.
        Args:
            bboxes: = shape(n, 4) the x1y1wh to real coordinate.
            scores: = shape(n,) the confidence.
            cls_ids: = shape(n,) the class id.
            image_id: int = image id.
            json_dt: list = a list to save prediction in coco json format.
        """
        for bbox, score, category_id in zip(bboxes.tolist(), scores.tolist(), cls_ids.tolist()):
            json_dt.append(

                {'image_id': int(image_id),
                 'category_id': int(category_id),
                 'bbox': [round(x, 3) for x in bbox],  # Keep 3 decimal places to get more accurate mAP
                 'score': round(score, 5)
                 }
            )

    @staticmethod
    def empty_append(json_dt: list):
        r"""
        Avoid bug that json_dt is empty when detect nothing.
        Args:
            json_dt: list = a list to save prediction in coco json format.
        """
        if not json_dt:
            json_dt.append(

                {'image_id': 0,
                 'category_id': 0,
                 'bbox': [0., 0., 0., 0.],
                 'score': 0
                 }
            )


class ReleaseMixin(object):
    r"""
    Methods:
        1. release_cuda_cache.
        2. release_attr.
    """

    @staticmethod
    def release_cuda_cache():
        r"""Release cuda cache"""
        torch.cuda.empty_cache()

    @staticmethod
    def release_attr(var=None):
        r"""Set variable None"""
        return var
