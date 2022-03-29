r"""
Meta Valer module for building all valer class.
"""

from utils.log import LOGGER, log_loss_and_metrics
from utils.metrics import compute_fps
from utils.mixins import ValDetectMixin

__all__ = ['MetaValDetect']


class MetaValDetect(ValDetectMixin):
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
        self.set_self_parameters_val_training(model, half, loss_fn, dataloader, cls_names, epoch, visual_image)

    def val(self):
        # TODO
        pass

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

    def set_self_parameters_val(self):
        # TODO
        # val alone
        pass

    def set_self_parameters_val_training(self, model, half, loss_fn, dataloader, cls_names, epoch, visual_image):
        # val during training
        if self.training:
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
