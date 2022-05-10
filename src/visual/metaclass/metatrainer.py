r"""
Meta Trainer module for building all trainer class.
"""

from ..utils.mixins import DataLoaderMixin, SetSavePathMixin, TrainMixin, \
    SaveCheckPointMixin, LoadAllCheckPointMixin, FreezeLayersMixin, COCOEvaluateMixin, \
    TensorboardWriterMixin, ReleaseMixin

__all__ = ['MetaTrainDetect']


class MetaTrainDetect(
    TrainMixin,
    ReleaseMixin,
    DataLoaderMixin,
    SetSavePathMixin,
    FreezeLayersMixin,
    COCOEvaluateMixin,
    SaveCheckPointMixin,
    TensorboardWriterMixin,
    LoadAllCheckPointMixin
):

    def train(self):
        raise NotImplementedError

    # @torch.no_grad()
    def val_training(self):
        raise NotImplementedError

    # @torch.inference_mode()
    def test_trained(self):
        raise NotImplementedError

    def visual_dataset(self, dataset_instance, name):
        raise NotImplementedError

# class MetaTrainClassify(
#     ReleaseMixin,
#     DataLoaderMixin,
#     SetSavePathMixin,
#     FreezeLayersMixin,
#     SaveCheckPointMixin,
#     TensorboardWriterMixin,
#     LoadAllCheckPointMixin
# ):
#     @logging_initialize('trainer')
#     def __init__(self, args):
#         super(MetaTrainClassify, self).__init__()
#         self.epoch = None
#         self.hyp = args.hyp
#         self.inc = args.inc
#         self.name = args.save_name
#         self.device = args.device
#         self.epochs = args.epochs
#         self.weights = args.weights
#         self.workers = args.workers
#         self.shuffle = args.shuffle
#         self.datasets = args.datasets
#         self.channels = args.channels
#         self.save_path = args.save_path
#         self.image_size = args.image_size
#         self.batch_size = args.batch_size
#         self.pin_memory = args.pin_memory
#         self.tensorboard = args.tensorboard
#         self.freeze_names = args.freeze_names
#         self.visual_image = args.visual_image
#         self.visual_graph = args.visual_graph
#
#         # Set load way
#         self._load_model = args.load_model
#         self._load_optimizer = args.load_optimizer
#         self._load_gradscaler = args.load_gradscaler
#         self._load_start_epoch = args.load_start_epoch
#         self._load_best_fitness = args.load_best_fitness
#         self._load_lr_scheduler = args.load_lr_scheduler
