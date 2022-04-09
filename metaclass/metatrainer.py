r"""
Meta Trainer module for building all trainer class.
"""

import torch

from utils.log import LOGGER, logging_initialize, logging_start_finish, log_loss
from utils.metrics import compute_fitness
from utils.mixins import DataLoaderMixin, SetSavePathMixin, TrainDetectMixin, \
    SaveCheckPointMixin, LoadAllCheckPointMixin, ResultsDealDetectMixin, FreezeLayersMixin, COCOEvaluateMixin, \
    TensorboardWriterMixin, TrainClassifyMixin

__all__ = ['MetaTrainDetect', 'MetaTrainClassify']


class MetaTrainDetect(
    DataLoaderMixin,
    SetSavePathMixin,
    TrainDetectMixin,
    FreezeLayersMixin,
    COCOEvaluateMixin,
    SaveCheckPointMixin,
    TensorboardWriterMixin,
    LoadAllCheckPointMixin,
    ResultsDealDetectMixin
):
    @logging_initialize('trainer')
    def __init__(self, args):
        super(MetaTrainDetect, self).__init__()
        # Get args for self.*
        self.epoch = None
        self.hyp = args.hyp
        self.inc = args.inc
        self.name = args.name
        self.device = args.device
        self.epochs = args.epochs
        self.weights = args.weights
        self.augment = args.augment
        self.workers = args.workers
        self.shuffle = args.shuffle
        self.datasets = args.datasets
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

        # Tensorboard must exist
        self.writer = None

        # todo Need to set in subclass __init__ below
        # Set val class
        self.val_class = None  # Must Need

        # To configure Trainer in subclass as following
        self.path_dict = None  # Get path_dict
        self.coco_eval = None  # (self.path_dict['json_gt'], self.path_dict['json_dt'])
        # self.device = select_one_device(self.device) # Set one device
        self.cuda = None  # For judging whether cuda
        # Load hyp yaml
        # Initialize or auto seed manual and save in self.hyp
        # Get datasets path dict
        # Save yaml dict
        # Empty args
        self.checkpoint = None  # Load checkpoint
        self.model = None  # Initialize or load model
        # Unfreeze model
        # Freeze layers of model
        self.param_groups = None  # Set parameter groups to for the optimizer
        self.optimizer = None  # Initialize and load optimizer
        self.lr_scheduler = None  # Initialize and load lr_scheduler
        self.scaler = None  # Initialize and load GradScaler
        self.start_epoch = None  # Initialize or load start_epoch
        self.best_fitness = None  # Initialize or load best_fitness
        # Empty self.checkpoint when load finished
        self.train_dataloader = None  # Get dataloader for training
        self.val_dataloader = None  # Get dataloader for validating
        self.test_dataloader = None  # Get dataloader for testing
        self.loss_fn = None  # Get loss function
        self.results = None  # To save results of training and validating

    @logging_start_finish('Training')
    def train(self):
        for self.epoch in range(self.start_epoch, self.epochs):
            loss = self.train_one_epoch()
            self.log_results(*loss)
            results_val = self.val_training()
            results, results_for_best = self.deal_results_memory(loss, results_val)
            self.add_data_results(('all_results', results[0]),
                                  ('all_class_results', results[1]))
            fitness = compute_fitness(results_for_best, self.hyp['fit_weights'])  # compute fitness for best save
            self.save_checkpoint(fitness)
            # TODO maybe need a auto stop function for bad training

        self.model = self.empty()
        test_results, coco_results = self.test_trained()
        results, _ = self.deal_results_memory((None, None), test_results)
        self.add_data_results(('all_results', results[0]),
                              ('all_class_results', results[1]))
        self.save_all_results()

        # save coco results
        self.save_coco_results(coco_results, self.path_dict['coco_results'])

        self.close_tensorboard()
        self.empty_cache()

    @torch.no_grad()
    def val_training(self):
        valer = self.val_class(last=False, model=self.model, half=True, dataloader=self.val_dataloader,
                               loss_fn=self.loss_fn, cls_names=self.datasets['names'],
                               epoch=self.epoch, writer=self.writer, visual_image=self.visual_image)
        results = valer.val_training()
        return results

    @torch.inference_mode()
    def test_trained(self):
        self.epoch = -1
        self.checkpoint = self.empty()
        self.checkpoint = self.load_checkpoint(self.path_dict['best'])
        if self.checkpoint is None:
            self.checkpoint = self.load_checkpoint(self.path_dict['last'])
            LOGGER.info('Load last.pt for validating because of no best.pt')
        else:
            LOGGER.info('Load best.pt for validating')
        self.model = self.load_model(load='model')
        if self.test_dataloader:
            dataloader = self.test_dataloader
        else:
            dataloader = self.val_dataloader
        tester = self.val_class(last=True, model=self.model, half=False, dataloader=dataloader,
                                loss_fn=self.loss_fn, cls_names=self.datasets['names'],
                                epoch=self.epoch, writer=self.writer, visual_image=self.visual_image,
                                coco_eval=self.coco_eval)
        results = tester.val_training()

        # get coco results
        coco_results = None
        if self.coco_eval:
            coco_results = self.coco_evaluate(dataloader, 'bbox')

        return results, coco_results

    def log_results(self, loss_all, loss_name):
        log_loss('Train', self.epoch, loss_name, loss_all)

    @staticmethod
    def empty_cache():
        r"""Empty cuda cache"""
        torch.cuda.empty_cache()

    @staticmethod
    def empty(empty=None):
        r"""Set variable None"""
        return empty


class MetaTrainClassify(
    DataLoaderMixin,
    SetSavePathMixin,
    FreezeLayersMixin,
    TrainClassifyMixin,
    SaveCheckPointMixin,
    TensorboardWriterMixin,
    LoadAllCheckPointMixin
):
    @logging_initialize('trainer')
    def __init__(self, args):
        super(MetaTrainClassify, self).__init__()
        self.epoch = None
        self.hyp = args.hyp
        self.inc = args.inc
        self.name = args.name
        self.device = args.device
        self.epochs = args.epochs
        self.weights = args.weights
        self.workers = args.workers
        self.shuffle = args.shuffle
        self.datasets = args.datasets
        self.channels = args.channels
        self.save_path = args.save_path
        self.image_size = args.image_size
        self.batch_size = args.batch_size
        self.pin_memory = args.pin_memory
        self.tensorboard = args.tensorboard
        self.freeze_names = args.freeze_names
        self.visual_image = args.visual_image
        self.visual_graph = args.visual_graph

        # Set load way
        self._load_model = args.load_model
        self._load_optimizer = args.load_optimizer
        self._load_gradscaler = args.load_gradscaler
        self._load_start_epoch = args.load_start_epoch
        self._load_best_fitness = args.load_best_fitness
        self._load_lr_scheduler = args.load_lr_scheduler

        # Tensorboard must exist
        self.writer = None

        # todo Need to set in subclass __init__ below
        # Set val class
        self.val_class = None  # Must Need

        # To configure Trainer in subclass as following
        self.path_dict = None  # Get path_dict
        # self.device = select_one_device(self.device) # Set one device
        self.cuda = None  # For judging whether cuda
        # Load hyp yaml
        # Initialize or auto seed manual and save in self.hyp
        # Get datasets path dict
        # Save yaml dict
        # Empty args
        self.checkpoint = None  # Load checkpoint
        self.model = None  # Initialize or load model
        # Unfreeze model
        # Freeze layers of model
        self.param_groups = None  # Set parameter groups to for the optimizer
        self.optimizer = None  # Initialize and load optimizer
        self.lr_scheduler = None  # Initialize and load lr_scheduler
        self.scaler = None  # Initialize and load GradScaler
        self.start_epoch = None  # Initialize or load start_epoch
        self.best_fitness = None  # Initialize or load best_fitness
        # Empty self.checkpoint when load finished
        self.train_dataloader = None  # Get dataloader for training
        self.val_dataloader = None  # Get dataloader for validating
        self.test_dataloader = None  # Get dataloader for testing
        self.loss_fn = None  # Get loss function
        self.results = None  # To save results of training and validating

    @logging_start_finish('Training')
    def train(self):
        for self.epoch in range(self.start_epoch, self.epochs):
            loss = self.train_one_epoch()
            self.log_results(*loss)
            results_val = self.val_training()
            self.save_checkpoint(results_val[1][0])
            # TODO maybe need a auto stop function for bad training

        self.model = self.empty()
        test_results = self.test_trained()

        self.close_tensorboard()
        self.empty_cache()

    @torch.no_grad()
    def val_training(self):
        valer = self.val_class(last=False, model=self.model, half=True, dataloader=self.val_dataloader,
                               loss_fn=self.loss_fn, cls_names=self.datasets['names'],
                               epoch=self.epoch, writer=self.writer, visual_image=self.visual_image)
        results = valer.val_training()
        return results

    @torch.inference_mode()
    def test_trained(self):
        self.epoch = -1
        self.checkpoint = self.empty()
        self.checkpoint = self.load_checkpoint(self.path_dict['best'])
        if self.checkpoint is None:
            self.checkpoint = self.load_checkpoint(self.path_dict['last'])
            LOGGER.info('Load last.pt for validating because of no best.pt')
        else:
            LOGGER.info('Load best.pt for validating')
        self.model = self.load_model(load='model')
        if self.test_dataloader:
            dataloader = self.test_dataloader
        else:
            dataloader = self.val_dataloader
        tester = self.val_class(last=True, model=self.model, half=False, dataloader=dataloader,
                                loss_fn=self.loss_fn, cls_names=self.datasets['names'],
                                epoch=self.epoch, writer=self.writer, visual_image=self.visual_image)
        results = tester.val_training()
        return results

    def log_results(self, loss_all, loss_name):
        log_loss('Train', self.epoch, loss_name, loss_all)

    @staticmethod
    def empty_cache():
        r"""Empty cuda cache"""
        torch.cuda.empty_cache()

    @staticmethod
    def empty(empty=None):
        r"""Set variable None"""
        return empty


if __name__ == '__main__':
    pass
