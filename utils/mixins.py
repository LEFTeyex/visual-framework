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

from utils import WRITER
from utils.log import LOGGER
from utils.check import check_only_one_set
from utils.bbox import xyxy2x1y1wh, rescale_xyxy
from utils.general import delete_list_indices, time_sync, loss_to_mean
from utils.typeslib import optimizer_, lr_scheduler_, gradscaler_, dataset_, \
    module_or_None, str_or_None, pkt_or_None, tuple_or_list, strpath

__all__ = ['SetSavePathMixin', 'TensorboardWriterMixin',
           'SaveCheckPointMixin', 'LoadAllCheckPointMixin',
           'FreezeLayersMixin', 'DataLoaderMixin',
           'TrainDetectMixin', 'ValDetectMixin',
           'TrainClassifyMixin', 'ValClassifyMixin',
           'COCOEvaluateMixin',
           'ReleaseMixin'
           ]


class SetSavePathMixin(object):
    r"""
    Methods:
        1. get_save_path --- need all self.*.
    """

    def __init__(self):
        self.save_name = None
        self.save_path = None

    def get_save_path(self, *args: tuple_or_list):
        r"""
        Set save path to path_dict.
        Args:
            args: tuple_or_list = (save_name, path), ...

        Returns:
            dict{'save_name': strpath, ...}
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
                        LOGGER.error(f'The {dir_name} is an unexpected save_name '
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
            path: strpath = StrPath to save tensorboard file
        """
        if self.tensorboard:
            LOGGER.info('Setting tensorboard writer...')
            writer = WRITER.set_writer(path)
            LOGGER.info('Setting tensorboard successfully')
            LOGGER.info(f"See tensorboard results please run "
                        f"'tensorboard --logdir={path}' in Terminal")
        else:
            LOGGER.info('No tensorboard writer')
            writer = None
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
        2. save_checkpoint --- need self.epoch, self.epochs, self.best_fitness.
    """

    def __init__(self):
        self.model = None
        self.epoch = None
        self.epochs = None
        self.scaler = None
        self.optimizer = None
        self.lr_scheduler = None
        self.best_fitness = None

    def get_checkpoint(self):
        r"""Get checkpoint to save"""
        LOGGER.debug('Getting checkpoint...')
        checkpoint = {
            'model': self.model,  # TODO de_parallel model in the future
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'gradscaler_state_dict': self.scaler.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict()
        }
        LOGGER.debug('Get checkpoint successfully')
        return checkpoint

    def save_checkpoint(self, fitness, best_path: strpath, last_path: strpath):
        r"""
        Save checkpoint when get better fitness or at last.
        Args:
            fitness: = a number to compare with best_fitness.
            best_path: strpath = StrPath to save best.pt.
            last_path: strpath = StrPath to save last.pt.
        """
        # save the best checkpoint
        if fitness > self.best_fitness:
            LOGGER.debug(f'Saving best checkpoint in epoch{self.epoch}...')
            checkpoint = self.get_checkpoint()
            self.best_fitness = fitness
            torch.save(checkpoint, best_path)
            LOGGER.debug(f'Save best checkpoint in epoch{self.epoch} successfully')

        # save the last checkpoint
        if self.epoch + 1 == self.epochs:
            LOGGER.info('Saving last checkpoint')
            checkpoint = self.get_checkpoint()
            torch.save(checkpoint, last_path)
            LOGGER.info('Save last checkpoint successfully')


class LoadAllCheckPointMixin(object):
    r"""
    Methods:
        1. load_checkpoint --- need self.device.
        2. load_model --- need self.device, self.checkpoint.
        3. load_optimizer --- need self.checkpoint.
        4. set_param_groups --- need self.model.
        5. load_lr_scheduler ---- need self.checkpoint.
        6. load_gradscaler --- need self.checkpoint.
        7. load_start_epoch --- need self.checkpoint, self.epochs.
        8. load_best_fitness --- need self.checkpoint.
    """

    def __init__(self):
        self.model = None
        self.device = None
        self.epochs = None
        self.checkpoint = None

    def load_checkpoint(self, path: strpath, suffix: tuple = ('.pt', '.pth')):
        r"""
        Load checkpoint from path '*.pt' or '*.pth' file.
        Request:
            checkpoint = {
                            'model': self.model,
                            'epoch': self.epoch,
                            'best_fitness': self.best_fitness,
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'gradscaler_state_dict': self.scaler.state_dict(),
                            'lr_scheduler_state_dict': self.lr_scheduler.state_dict()
                         }

        Args:
            path: strpath = StrPath of checkpoint.
            suffix: tuple = ('.pt', '.pth', ...) the suffix of checkpoint file.

        Returns:
            checkpoint or None(when no file endwith the suffix)
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

    def load_model(self, model_instance: module_or_None = None, load: str_or_None = 'state_dict'):
        r"""
        Load model from 'model' or 'state_dict' or initialize model.
        Args:
            model_instance: module_or_None = ModelDetect() the instance of model, Default=None(only when load='model').
            load: str_or_None = None / 'model' / 'state_dict', Default='state_dict'(load model from state_dict).

        Returns:
            model_instance in float
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
            missing_keys, unexpected_keys = rest
            if missing_keys or unexpected_keys:
                LOGGER.warning(f'There are the rest of {len(missing_keys)} missing_keys'
                               f' and {len(unexpected_keys)} unexpected_keys when load model')
                LOGGER.warning(f'missing_keys: {missing_keys}')
                LOGGER.warning(f'unexpected_keys: {unexpected_keys}')
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

    def load_optimizer(self, optim_instance: optimizer_, load: bool = True):
        r"""
        Load optimizer from state_dict and add param_groups first.
        Args:
            optim_instance: optimizer_ = SGD(self.model.parameters(), lr=0.01) etc. the instance of optimizer.
            load: bool = True / False, Default=True(load optimizer state_dict).

        Returns:
            optim_instance
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

    def set_param_groups(self, param_kind_tuple: pkt_or_None = None):
        r"""
        Set param_groups, need to know how to set param_kind_list.
        The parameters in param_groups will not repeat.
        Args:
            param_kind_tuple: pkt_or_None = (('bias', nn.Parameter, {lr=0.01}),
                                              ('weight', nn.BatchNorm2d, {lr=0.02})).
                                             Default=None(do not set param_groups).

        Returns:
            param_groups [{'params': nn.Parameter, 'lr': 0.01, ...}, ...]
        """
        # do not set param_groups
        if param_kind_tuple is None:
            LOGGER.info('No param_groups set')
            return
        # set param_groups
        LOGGER.info('Setting param_groups...')
        # TODO Upgrade the algorithm in the future for filtering parameters from model.modules() better
        param_groups = []
        rest = []  # save_params the rest parameters that are not filtered
        indices = []  # save_params parameters (save_name, ...) temporarily to delete

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
            param_dict[
                'save_name'] = f'{kind.__name__}.{name}'  # to record in tensorboard for lr corresponding to params
            param_groups.append(param_dict)

        # check whether rest is empty
        if rest:
            LOGGER.exception(f'There are still {len(rest)} modules ungrouped,'
                             f' please reset the arg param_kind_tuple')
            for module, param_kind in (rest, indices):
                LOGGER.info(f'{module} --- {param_kind}')

        LOGGER.info('Set param_groups successfully')
        return param_groups

    def load_lr_scheduler(self, lr_scheduler_instance: lr_scheduler_, load: bool = True):
        r"""
        Load lr_scheduler from state_dict.
        Args:
            lr_scheduler_instance: lr_scheduler_ = StepLR(self.optimizer, 30) etc. the instance of lr_scheduler.
            load: bool = True / False, Default=True(load lr_scheduler state_dict).

        Returns:
            scheduler_instance
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

    def load_gradscaler(self, gradscaler_instance: gradscaler_, load: bool = True):
        r"""
        Load GradScaler from state_dict.
        Args:
            gradscaler_instance: gradscaler_ = GradScaler(enabled=self.cuda) etc. the instance of GradScaler.
            load: bool = True / False, Default=True(load GradScaler state_dict).

        Returns:
            gradscaler_instance
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

    def load_start_epoch(self, load: str_or_None = 'continue'):
        r"""
        Load start_epoch.
        Args:
            load: str_or_None = 'continue' / 'add' / None, Default='continue'.
                continue: continue to train from start_epoch to self.epochs.
                add: train self.epochs more times.
                None: initialize start_epoch=0.

        Returns:
            start_epoch
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
            load: bool = True / False, Default=True(load best_fitness).

        Returns:
            best_fitness
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
            best_fitness = 0.0
            LOGGER.info(f'Initialize best_fitness={best_fitness} successfully')
        return best_fitness

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
            layer_names: list = list consist of save_name in model layers.
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
            layer_names: list = list consist of save_name in model layers.
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
    r"""
    Methods:
        1. set_dataloader --- need all self.*.
    """

    def __init__(self):
        self.writer = None
        self.workers = None
        self.batch_size = None
        self.pin_memory = None
        self.visual_image = None

    def set_dataloader(self, dataset_instance: dataset_, shuffle: bool = False):
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
            LOGGER.info(f'Visualizing Dataset...')
            name = dataset_instance.name
            WRITER.add_datasets_images_labels_detect(self.writer, dataset_instance, name)
            LOGGER.info(f'Visualize Dataset successfully')

        # set dataloader
        # TODO upgrade num_workers(deal and check) and sampler(distributed.DistributedSampler) for DDP
        if hasattr(dataset_instance, 'collate_fn'):
            collate_fn = dataset_instance.collate_fn
        else:
            collate_fn = None
        batch_size = min(self.batch_size, len(dataset_instance))

        dataloader = DataLoader(dataset_instance, batch_size, shuffle,
                                num_workers=self.workers,
                                pin_memory=self.pin_memory,
                                collate_fn=collate_fn)
        LOGGER.info(f'Initialize Dataloader successfully')
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


class _TrainMixin(object):
    r"""
    Consist of basic methods for training.
    Methods:
        1. optimize_no_scale --- self.optimizer.
        2. optimize_scale --- self.scaler, self.optimizer.
        3. show_loss_in_pbar --- self.epoch, self.epochs.
    """

    def __init__(self):
        self.epoch = None
        self.epochs = None
        self.scaler = None
        self.optimizer = None

    def optimize_no_scale(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def optimize_scale(self, loss):
        self.scaler.scale(loss).backward()
        # todo maybe accumulating gradient will be better (if index...: ...)
        self.scaler.step(self.optimizer)  # optimizer.step()
        self.scaler.update()
        self.optimizer.zero_grad()  # improve a little performance when set_to_none=True

    def show_loss_in_pbar(self, loss, loss_name, pbar):
        # GPU memory used which an accurate value because of 1024 * 1024 * 1024 = 1073741824
        memory_cuda = f'GPU: {torch.cuda.memory_reserved() / 1073741824 if torch.cuda.is_available() else 0:.3f}GB'

        # show in pbar
        space = ' ' * 11
        progress = f'{self.epoch}/{self.epochs - 1}:'
        pbar.set_description_str(f"{space}epoch {progress:<9}{memory_cuda}")
        show = ''.join([f'{x}: {y:.3f} ' for x, y in zip(loss_name, loss)])
        pbar.set_postfix_str(show)


class _ValMixin(object):
    r"""
    Consist of basic methods for validating.
    Methods:
        1. show_loss_in_pbar.
    """

    @staticmethod
    def show_loss_in_pbar(loss, loss_name, pbar):
        # GPU memory used which an accurate value because of 1024 * 1024 * 1024 = 1073741824
        memory_cuda = f'GPU: {torch.cuda.memory_reserved() / 1073741824 if torch.cuda.is_available() else 0:.3f}GB'

        # show in pbar
        space = ' ' * 11
        pbar.set_description_str(f"{space}{'validating:':<15}{memory_cuda}")
        show = ''.join([f'{x}: {y:.3f} ' for x, y in zip(loss_name, loss)])
        pbar.set_postfix_str(show)


class TrainDetectMixin(_TrainMixin):
    r"""
    Methods:
        1. train_one_epoch --- need self.* except self.epochs.
    """

    def __init__(self):
        super(TrainDetectMixin, self).__init__()
        self.inc = None
        self.cuda = None
        self.epoch = None
        self.model = None
        self.device = None
        self.scaler = None
        self.writer = None
        self.loss_fn = None
        self.optimizer = None
        self.image_size = None
        self.visual_graph = None
        self.lr_scheduler = None
        self.train_dataloader = None

    def train_one_epoch(self, loss_name: tuple_or_list):
        r"""
        Finish train one epoch.
        Args:
            loss_name: tuple_or_list = the name of loss corresponding to the loss from loss_fn.
        """
        self.model.train()
        if self.visual_graph:
            WRITER.add_model_graph(self.writer, self.model, self.inc, self.image_size, self.epoch)

        loss_mean = torch.tensor(0., device=self.device)

        with tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader),
                  bar_format='{l_bar}{bar:10}{r_bar}') as pbar:
            for index, (images, labels, _, _) in pbar:
                images = images.to(self.device).float() / 255  # to float32 and normalized 0.0-1.0
                labels = labels.to(self.device)

                # TODO warmup in the future

                # forward with mixed precision
                with autocast(enabled=self.cuda):
                    outputs = self.model(images)
                    loss, loss_items = self.loss_fn(outputs, labels)

                # backward and optimize
                if self.scaler is None:
                    self.optimize_no_scale(loss)
                else:
                    self.optimize_scale(loss)

                # mean total loss and loss items
                loss_all = torch.cat((loss, loss_items), dim=0).detach()
                loss_mean = loss_to_mean(index, loss_mean, loss_all)
                self.show_loss_in_pbar(loss_mean, loss_name, pbar)

            WRITER.add_optimizer_lr(self.writer, self.optimizer, self.epoch)
            WRITER.add_epoch_curve(self.writer, 'train_loss', loss_mean, loss_name, self.epoch)

            self.lr_scheduler.step()


class TrainClassifyMixin(_TrainMixin):
    def __init__(self):
        super(TrainClassifyMixin, self).__init__()
        self.inc = None
        self.cuda = None
        self.model = None
        self.epoch = None
        self.scaler = None
        self.device = None
        self.writer = None
        self.epochs = None
        self.loss_fn = None
        self.optimizer = None
        self.image_size = None
        self.lr_scheduler = None
        self.visual_graph = None
        self.train_dataloader = None

    def train_one_epoch(self):
        self.model.train()
        if self.visual_graph:
            WRITER.add_model_graph(self.writer, self.model, self.inc, self.image_size, self.epoch)

        loss_name = ('class_loss',)
        loss_mean = torch.zeros(1, device=self.device)

        with tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader),
                  bar_format='{l_bar}{bar:10}{r_bar}') as pbar:
            for index, (images, labels) in pbar:
                # images = images.to(self.device).float() / 255  # to float32 and normalized 0.0-1.0
                images = images.to(self.device)
                labels = labels.to(self.device)

                # forward with mixed precision
                with autocast(enabled=self.cuda):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)

                # backward and optimize
                if self.scaler is None:
                    self.optimize_no_scale(loss)
                else:
                    self.optimize_scale(loss)

                _loss = loss.detach()
                loss_mean = loss_to_mean(index, loss_mean, _loss)
                self.show_loss_in_pbar(loss_mean, loss_name, pbar)

            WRITER.add_optimizer_lr(self.writer, self.optimizer, self.epoch)
            WRITER.add_epoch_curve(self.writer, 'train_loss', loss_mean, loss_name, self.epoch)

            # lr_scheduler
            self.lr_scheduler.step()
        return loss_mean.tolist(), loss_name


class ValDetectMixin(_ValMixin):
    r"""
    Methods:
        1. val_once --- need all self.*.
    """

    def __init__(self):
        self.hyp = None
        self.time = None
        self.seen = None
        self.half = None
        self.model = None
        self.epoch = None
        self.writer = None
        self.device = None
        self.loss_fn = None
        self.coco_json = None
        self.dataloader = None
        self.visual_image = None

    def val_once(self, loss_name: tuple_or_list):
        r"""
        Finish val once.
        Args:
            loss_name: tuple_or_list = the name of loss corresponding to the loss from loss_fn.

        Returns:
            json_dt list
        """
        loss_mean = torch.tensor(0., device=self.device).half() if self.half else torch.tensor(0., device=self.device)
        json_dt = []  # save detection truth for COCO eval
        with tqdm(enumerate(self.dataloader), total=len(self.dataloader),
                  bar_format='{l_bar:>42}{bar:10}{r_bar}') as pbar:
            for index, (images, labels, shape_converts, img_ids) in pbar:
                # get current for computing FPs but maybe not accurate maybe
                t0 = time_sync()
                images = images.to(self.device)
                labels = labels.to(self.device)

                # to half16 or float32 and normalized 0.0-1.0
                images = images.half() / 255 if self.half else images.float() / 255

                # inference
                predictions = self.model(images)
                loss, loss_items = self.loss_fn(predictions, labels)

                # mean total loss and loss items
                loss_all = torch.cat((loss, loss_items), dim=0).detach()
                loss_mean = loss_to_mean(index, loss_mean, loss_all)
                self.show_loss_in_pbar(loss_mean, loss_name, pbar)

                # parse outputs to predictions bbox is xyxy
                predictions = self.model.decode(predictions,
                                                self.hyp['obj_threshold'],
                                                self.hyp['iou_threshold'],
                                                self.hyp['max_detect'])

                self.time += time_sync() - t0

                if self.visual_image:
                    WRITER.add_batch_images_predictions_detect(self.writer, 'test_pred', index, images, predictions,
                                                               self.epoch)

                # add metrics data to json_dt
                # TODO BUG the predictions maybe empty 2022.4.19
                for idx, (p, img_id) in enumerate(zip(predictions, img_ids)):
                    self.seen += 1
                    p[:, :4] = rescale_xyxy(p[:, :4], shape_converts[idx])  # to original image shape
                    p[:, :4] = xyxy2x1y1wh(p[:, :4])
                    COCOEvaluateMixin.append_json_dt(p, img_id, json_dt)

            WRITER.add_epoch_curve(self.writer, 'val_loss', loss_mean, loss_name, self.epoch)

        if self.epoch == -1:  # save json_dt in the test
            with open(self.coco_json[1], 'w') as f:
                json.dump(json_dt, f)

        return json_dt


class ValClassifyMixin(_ValMixin):
    def __init__(self):
        self.time = None
        self.seen = None
        self.half = None
        self.model = None
        self.epoch = None
        self.writer = None
        self.device = None
        self.loss_fn = None
        self.dataloader = None
        self.visual_image = None

    def val_once(self):
        loss_name = ('class_loss',)
        loss_mean = torch.zeros(1, device=self.device).half() if self.half else torch.zeros(1, device=self.device)
        stats = []
        with tqdm(enumerate(self.dataloader), total=len(self.dataloader),
                  bar_format='{l_bar:>42}{bar:10}{r_bar}') as pbar:
            for index, (images, labels) in pbar:
                # get current for computing FPs but maybe not accurate maybe
                t0 = time_sync()
                images = images.to(self.device)
                labels = labels.to(self.device)

                # to half16 or float32 and normalized 0.0-1.0
                # images = images.half() / 255 if self.half else images.float() / 255
                images = images.half() if self.half else images.float()

                # inference
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                # mean total loss and loss items
                _loss = loss.detach()
                loss_mean = loss_to_mean(index, loss_mean, _loss)
                self.show_loss_in_pbar(loss_mean, loss_name, pbar)

                self.time += time_sync() - t0
                self.seen += images.shape[0]
                self._get_metrics_stats(outputs, labels, stats)

                if self.visual_image:
                    # TODO add image without label for classification
                    pass

            WRITER.add_epoch_curve(self.writer, 'val_loss', loss_mean, loss_name, self.epoch)

        return loss_mean.tolist(), loss_name, stats

    def compute_metrics(self, stats):
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        nc = self.model.nc
        pre, conf, labels = stats
        cls_number = np.bincount(labels.astype(np.int64), minlength=nc)
        cls_top1_number = np.zeros_like(cls_number)
        for cls in range(nc):
            filter_cls = labels == cls
            top1_number = np.sum(pre[filter_cls] == labels[filter_cls]).tolist()
            cls_top1_number[cls] = top1_number
        top1 = (cls_top1_number.sum() / cls_number.sum()).tolist()
        top1_cls = (cls_top1_number / cls_number).tolist()

        fmt = '<10.3f'
        space = ' ' * 50
        LOGGER.info(f'{space}top1: {top1:{fmt}}')
        return top1, top1_cls, cls_number

    @staticmethod
    def _get_metrics_stats(predictions, labels, stats):
        cls_conf, cls_pre = torch.max(predictions, dim=1)
        stats.append((cls_pre.cpu(), cls_conf.cpu(), labels.cpu()))


class COCOEvaluateMixin(object):
    r"""
    Methods:
        1. coco_evaluate
        2. save_coco_results
        3. append_json_dt
    """

    @staticmethod
    def coco_evaluate(coco_gt: strpath, coco_dt, img_ids: list, eval_type: str = 'bbox'):
        r"""
        Evaluate by coco.
        Args:
            coco_gt: strpath = StrPath of coco_gt json.
            coco_dt: = StrPath of coco_dt json / list of coco_dt.
            img_ids: list = image id to evaluate.
            eval_type: str = evaluate type consist of ('segm', 'bbox', 'keypoints').

        Returns:
            coco_results
        """

        if not Path(coco_gt).exists():
            raise FileExistsError(f'The coco_gt {coco_gt} json file do not exist')

        if not isinstance(coco_dt, list):
            if not Path(coco_gt).exists():
                raise FileExistsError(f'The coco_dt {coco_dt} json file do not exist')

        coco_gt = COCO(coco_gt)
        coco_dt = coco_gt.loadRes(coco_dt)
        coco = COCOeval(coco_gt, coco_dt, iouType=eval_type)
        coco.params.imgIds = img_ids
        coco.evaluate()
        coco.accumulate()
        coco.summarize()
        coco_results = coco.eval
        return coco_results

    @staticmethod
    def save_coco_results(coco_results, path: strpath):
        r"""
        Save coco_results.
        Args:
            coco_results: = coco.eval.
            path: strpath = save path.
        """
        path = str(path)
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

    @staticmethod
    def append_json_dt(predictions, image_id: int, json_dt: list):
        r"""
        Append predictions to json_dt list.
        Args:
            predictions: = list of prediction ( n, 6(x1, y1, w, h, conf, cls_idx) ) corresponding to original label.
            image_id: int = image id.
            json_dt: list = save prediction in coco json format.
        """
        bboxes = predictions[:, :4]
        scores = predictions[:, 4]
        cls_id = predictions[:, 5]
        for category_id, bbox, score in zip(cls_id.tolist(), bboxes.tolist(), scores.tolist()):
            json_dt.append({'image_id': int(image_id),
                            'category_id': int(category_id),
                            'bbox': [round(x, 3) for x in bbox],  # Keep 3 decimal places to get more accurate mAP
                            'score': float(score)})


class ReleaseMixin(object):
    @staticmethod
    def release_cuda_cache():
        r"""Release cuda cache"""
        torch.cuda.empty_cache()

    @staticmethod
    def release(var=None):
        r"""Set variable None"""
        return var


if __name__ == '__main__':
    pass
