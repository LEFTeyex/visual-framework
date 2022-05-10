r"""
Meta Valer module for building all valer class.
"""

from ..utils.mixins import ValMixin, SetSavePathMixin, LoadAllCheckPointMixin, DataLoaderMixin, \
    COCOEvaluateMixin, ReleaseMixin

__all__ = ['MetaValDetect']


class MetaValDetect(
    ValMixin,
    ReleaseMixin,
    DataLoaderMixin,
    SetSavePathMixin,
    COCOEvaluateMixin,
    LoadAllCheckPointMixin
):

    # @torch.no_grad()
    def val_training(self):
        raise NotImplementedError

    # @torch.inference_mode()
    def val(self):
        raise NotImplementedError

    def visual_dataset(self, dataset_instance, name):
        raise NotImplementedError

# class MetaValClassify(ValClassifyMixin):
#
#     def __init__(self, last=True, model=None, writer=None,  # need to super in subclass
#                  half=True, dataloader=None, loss_fn=None, cls_names=None, epoch=None, visual_image=None,
#                  args=None):
#         super(MetaValClassify, self).__init__()
#         self.last = last
#         self.seen = 0
#         self.time = 0.0
#         self.writer = writer
#         self.training = model is not None
#         self.cls_names = None
#         if self.training:
#             self.set_self_parameters_val_training(model, half, loss_fn, dataloader, cls_names, epoch, visual_image)
#         else:
#             pass
#
#     def val(self):
#         pass
#
#     @torch.no_grad()
#     def val_training(self):
#         self.model.eval()
#         loss, loss_name, stats = self.val_once()
#         metrics = self.compute_metrics(stats)
#         fps_time = compute_fps(self.seen, self.time)
#         self.log_results(loss, loss_name, metrics, fps_time)
#         self.model.float()
#         return (loss, loss_name), metrics, fps_time
#
#     def log_results(self, loss, loss_name, metrics, fps_time):
#         t_fmt = '<15'  # title format
#         fmt = t_fmt + '.3f'
#         space = ' ' * 50
#         WRITER.add_epoch_curve(self.writer, 'val_metrics', (metrics[0],), ('top1',), self.epoch)
#         LOGGER.debug(f'{self.epoch}, {loss_name}, {loss}, {metrics}, {fps_time}')
#         if self.last:
#             LOGGER.info(f'{space}Speed {fps_time[1]:.2f} ms per image, FPs: {fps_time[0]:.1f}, accuracy')
#         else:
#             LOGGER.info(f'{space}Speed {fps_time[1]:.2f} ms per image, FPs: {fps_time[0]:.1f}, no accuracy')
#         if self.last:
#             LOGGER.info(f"{'class_name':{t_fmt}}"
#                         f"{'number':{t_fmt}}"
#                         f"{'top1':{t_fmt}}")
#             for name, num, top1 in zip(self.cls_names, metrics[2], metrics[1]):
#                 LOGGER.info(f'{name:{t_fmt}}'
#                             f'{num:{fmt}}'
#                             f'{top1:{fmt}}')
#
#     def set_self_parameters_val_training(self, model, half, loss_fn, dataloader, cls_names, epoch, visual_image):
#         # val during training
#         self.device = next(model.parameters()).device
#         self.half = half
#         self.epoch = epoch
#         self.loss_fn = loss_fn
#         self.cls_names = cls_names
#         self.visual_image = visual_image
#         self.dataloader = dataloader
#         if self.half and self.device.type == 'cpu':
#             LOGGER.warning(f'The device is {self.device}, half precision only supported on CUDA')
#             self.half = False
#         self.model = model.half() if self.half else model.float()
