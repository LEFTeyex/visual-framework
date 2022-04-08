r"""
Meta Valer module for building all valer class.
"""

import torch

from utils.log import LOGGER, log_loss_and_metrics
from utils.metrics import compute_fps
from utils.mixins import ValDetectMixin, SetSavePathMixin, LoadAllCheckPointMixin, DataLoaderMixin, \
    COCOEvaluateMixin

__all__ = ['MetaValDetect']


class MetaValDetect(
    ValDetectMixin,
    DataLoaderMixin,
    SetSavePathMixin,
    COCOEvaluateMixin,
    LoadAllCheckPointMixin
):
    def __init__(self, last=True, model=None, writer=None,  # need to super in subclass
                 half=True, dataloader=None, loss_fn=None, cls_names=None, epoch=None, visual_image=None,
                 coco_eval=None,  # coco_eval is the tuple with path of gt and dt json file
                 args=None):
        super(MetaValDetect, self).__init__()
        self.last = last
        self.seen = 0
        self.time = 0.0
        self.writer = writer
        self.training = model is not None
        self.cls_names = None
        self.coco_eval = coco_eval
        if self.training:
            self.set_self_parameters_val_training(model, half, loss_fn, dataloader, cls_names, epoch, visual_image)
        else:
            self.inc = None
            self.hyp = None
            self.task = None
            self.name = None
            self.half = None
            self.device = None
            self.weights = None
            self.workers = None
            self.datasets = None
            self.save_path = None
            self.batch_size = None
            self.pin_memory = None
            self.image_size = None

            self.tensorboard = None  # TODO a bug about super in subclass
            self.visual_image = None  # TODO a bug about super in subclass
            self.epoch = -1  # TODO a bug about super in subclass

            self.model = None
            self.loss_fn = None
            self.cls_names = None
            self.path_dict = None
            self.checkpoint = None
            self.dataloader = None
            self.set_self_parameters_val(args)

    @torch.inference_mode()
    def val(self):
        self.model.eval()
        loss_all, loss_name, stats = self.val_once()
        metrics = self.compute_metrics(stats)
        fps_time = compute_fps(self.seen, self.time)
        self.log_results(loss_all, loss_name, metrics, fps_time)
        coco_results = self.coco_evaluate(self.dataloader, 'bbox')
        self.save_coco_results(coco_results, self.path_dict['coco_results'])
        self.empty_cache()

    @torch.no_grad()
    def val_training(self):
        self.model.eval()
        # TODO maybe save something or plot images below
        loss_all, loss_name, stats = self.val_once()
        metrics = self.compute_metrics(stats)
        fps_time = compute_fps(self.seen, self.time)
        # TODO confusion matrix needed
        # TODO get the stats of the target number per class which detected correspond to label correctly
        self.log_results(loss_all, loss_name, metrics, fps_time)
        if not self.last:
            self.model.float()
        return (loss_all, loss_name), metrics, fps_time

    def log_results(self, loss_all, loss_name, metrics, fps_time):
        log_loss_and_metrics('Val', self.epoch, self.last, self.writer, self.cls_names,
                             loss_name, loss_all, metrics, fps_time)

    def set_self_parameters_val(self, args):
        r"""Implemented in subclass and super"""
        self.inc = args.inc
        self.hyp = args.hyp
        self.task = args.task
        self.name = args.name
        self.half = args.half
        self.device = args.device
        self.weights = args.weights
        self.workers = args.workers
        self.datasets = args.datasets
        self.save_path = args.save_path
        self.image_size = args.image_size
        self.batch_size = args.batch_size
        self.pin_memory = args.pin_memory

    def set_self_parameters_val_training(self, model, half, loss_fn, dataloader, cls_names, epoch, visual_image):
        # val during training
        self.device = next(model.parameters()).device
        self.half = half
        self.epoch = epoch
        self.loss_fn = loss_fn
        self.cls_names = cls_names
        self.visual_image = visual_image
        self.dataloader = dataloader
        if self.half and self.device.type == 'cpu':
            LOGGER.warning(f'The device is {self.device}, half precision only supported on CUDA')
            self.half = False
        self.model = model.half() if self.half else model.float()

    @staticmethod
    def empty_cache():
        r"""Empty cuda cache"""
        torch.cuda.empty_cache()
