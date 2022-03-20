r"""
Meta Trainer module for building all trainer class.
"""

import torch

from pathlib import Path

from utils.log import LOGGER, logging_initialize, logging_start_finish, log_loss
from utils.mixins import LossMixin, DataLoaderMixin, SetSavePathMixin, TrainDetectMixin, \
    SaveCheckPointMixin, LoadAllCheckPointMixin, ResultsDealDetectMixin

__all__ = ['MetaTrainDetect']


class MetaTrainDetect(
    LossMixin,
    DataLoaderMixin,
    SetSavePathMixin,
    TrainDetectMixin,
    SaveCheckPointMixin,
    LoadAllCheckPointMixin,
    ResultsDealDetectMixin,
):
    @logging_initialize('trainer')
    def __init__(self, args):
        super(MetaTrainDetect, self).__init__()
        # # example
        # # Get args for self.*
        # self.hyp = None
        # self.inc = None
        # self.name = None
        # self.epoch = None
        # self.device = None
        # self.epochs = None
        # self.augment = None
        # self.workers = None
        # self.shuffle = None
        # self.weights = None
        # self.datasets = None
        # self.save_path = None
        # self.image_size = None
        # self.batch_size = None
        # self.pin_memory = None
        # self.tensorboard = None
        # self.data_augment = None

        # # Set load way
        # self._load_model = None
        # self._load_optimizer = None
        # self._load_gradscaler = None
        # self._load_start_epoch = None
        # self._load_best_fitness = None
        # self._load_lr_scheduler = None

        self.epoch = None
        self.hyp = args.hyp
        self.inc = args.inc
        self.name = args.name
        self.device = args.device
        self.epochs = args.epochs
        self.augment = args.augment
        self.workers = args.workers
        self.shuffle = args.shuffle
        self.datasets = args.datasets
        self.weights = Path(args.weights)
        self.image_size = args.image_size
        self.batch_size = args.batch_size
        self.pin_memory = args.pin_memory
        self.tensorboard = args.tensorboard
        self.save_path = Path(args.save_path)
        self.visual_image = args.visual_image
        self.visual_graph = args.visual_graph
        self.data_augment = args.data_augment

        self._load_model = args.load_model
        self._load_optimizer = args.load_optimizer
        self._load_gradscaler = args.load_gradscaler
        self._load_start_epoch = args.load_start_epoch
        self._load_best_fitness = args.load_best_fitness
        self._load_lr_scheduler = args.load_lr_scheduler

        # Tensorboard must exist
        self.writer = None

        # todo using Need to set in subclass __init__ below
        # Set val class
        self.val_class = None  # Must Need

        # To configure Trainer in subclass as following
        self.save_dict = None  # Get save_dict
        # self.device = None  # Set one device
        self.cuda = None  # For judging whether cuda
        # Load hyp yaml
        # Initialize or auto seed manual and save in self.hyp
        # Get datasets path dict
        # Save yaml dict
        # Empty args
        self.checkpoint = None  # Load checkpoint
        self.model = None  # Initialize or load model
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
            self.save_checkpoint(results_for_best)
            # TODO maybe need a auto stop function for bad training

        self.model = self.empty()
        test_results = self.test_trained()
        results, _ = self.deal_results_memory((None, None), test_results)
        self.add_data_results(('all_results', results[0]),
                              ('all_class_results', results[1]))
        self.save_all_results()
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
        tester = self.val_class(last=True, model=self.model, half=False, dataloader=dataloader,
                                loss_fn=self.loss_fn, cls_names=self.datasets['names'],
                                epoch=self.epoch, writer=self.writer, visual_image=self.visual_image)
        results = tester.val_training()
        return results

    def close_tensorboard(self):
        r"""Close writer which is the instance of SummaryWriter in tensorboard"""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

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


r"""def demo_parse_args_detect(known: bool = False):
        '''
        Parse args for training.
        Args:
            known: bool = True or False, Default=False
                parser will get two namespace which the second is unknown args, if known=True.
    
        Return namespace(for setting args)
        '''
        parser = argparse.ArgumentParser()
        parser.add_argument('--tensorboard', type=bool, default=True, help='')
        parser.add_argument('--visual_image', type=bool, default=True,
                            help='whether make images visual in tensorboard')
        parser.add_argument('--visual_graph', type=bool, default=False,
                            help='whether make model graph visual in tensorboard')
        parser.add_argument('--weights', type=str, default=str(ROOT / ''), help='')
        parser.add_argument('--device', type=str, default='0', help='cpu or cuda:0 or 0')
        parser.add_argument('--epochs', type=int, default=300, help='epochs for training')
        parser.add_argument('--batch_size', type=int, default=2, help='')
        parser.add_argument('--workers', type=int, default=0, help='')
        parser.add_argument('--shuffle', type=bool, default=True, help='')
        parser.add_argument('--pin_memory', type=bool, default=True, help='')
        parser.add_argument('--datasets', type=str, default=str(ROOT / 'data/datasets/Mydatasets.yaml'), help='')
        parser.add_argument('--name', type=str, default='exp', help='')
        parser.add_argument('--save_path', type=str, default=str(ROOT / 'runs/train'), help='')
        parser.add_argument('--hyp', type=str, default=str(ROOT / 'data/hyp/hyp_detect_train.yaml'), help='')
        parser.add_argument('--augment', type=bool, default=False, help='whether random augment image')
        parser.add_argument('--data_augment', type=str, default='mosaic',
                            help='the kind of data augmentation mosaic / mixup / cutout')
        parser.add_argument('--inc', type=int, default=3, help='')
        parser.add_argument('--image_size', type=int, default=640, help='')
        parser.add_argument('--load_model', type=str, default=None, help='')
        parser.add_argument('--load_optimizer', type=bool, default=False, help='')
        parser.add_argument('--load_lr_scheduler', type=bool, default=False, help='')
        parser.add_argument('--load_gradscaler', type=bool, default=False, help='')
        parser.add_argument('--load_start_epoch', type=str, default=None, help='')
        parser.add_argument('--load_best_fitness', type=bool, default=False, help='')
        namespace = parser.parse_known_args()[0] if known else parser.parse_args()
        return namespace"""

if __name__ == '__main__':
    pass
