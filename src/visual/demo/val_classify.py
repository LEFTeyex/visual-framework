# r"""
# For validating model.
# Consist of some Valers.
# """
#
# from pathlib import Path
#
# from metaclass.metavaler import MetaValClassify
#
# __all__ = ['ValClassify']
#
# r"""Set Global Constant for file save and load"""
# ROOT = Path.cwd()  # **/visual-framework root directory
#
#
# class ValClassify(MetaValClassify):
#     def __init__(self, args=None,
#                  last=True, model=None, writer=None,
#                  half=True, dataloader=None, loss_fn=None, cls_names=None, epoch=None, visual_image=None):
#         super(ValClassify, self).__init__(last, model, writer, half, dataloader,
#                                           loss_fn, cls_names, epoch, visual_image, args)
#
#     @logging_start_finish('Training')
#     def train(self):
#         for self.epoch in range(self.start_epoch, self.epochs):
#             loss = self.train_one_epoch()
#             results_val = self.val_training()
#             self.save_checkpoint(results_val[1][0], self.path_dict['best'], self.path_dict['last'])
#
#         self.model = self.release()
#         test_results = self.test_trained()
#
#         self.close_tensorboard()
#         self.release_cuda_cache()
#
#     @torch.no_grad()
#     def val_training(self):
#         valer = self.val_class(last=False, model=self.model, half=True, dataloader=self.val_dataloader,
#                                loss_fn=self.loss_fn, cls_names=self.datasets['names'],
#                                epoch=self.epoch, writer=self.writer, visual_image=self.visual_image)
#         results = valer.val_training()
#         return results
#
#     # @torch.inference_mode()
#     @torch.no_grad()
#     def test_trained(self):
#         self.epoch = -1
#         self.checkpoint = self.release()
#         self.checkpoint = self.load_checkpoint(self.path_dict['best'])
#         if self.checkpoint is None:
#             self.checkpoint = self.load_checkpoint(self.path_dict['last'])
#             LOGGER.info('Load last.pt for validating because of no best.pt')
#         else:
#             LOGGER.info('Load best.pt for validating')
#         self.model = self.load_model(load='model')
#         if self.test_dataloader:
#             dataloader = self.test_dataloader
#         else:
#             dataloader = self.val_dataloader
#         tester = self.val_class(last=True, model=self.model, half=False, dataloader=dataloader,
#                                 loss_fn=self.loss_fn, cls_names=self.datasets['names'],
#                                 epoch=self.epoch, writer=self.writer, visual_image=self.visual_image)
#         results = tester.val_training()
#         return results
#
#     def compute_metrics(self, stats):
#         stats = [np.concatenate(x, 0) for x in zip(*stats)]
#         nc = self.model.nc
#         pre, conf, labels = stats
#         cls_number = np.bincount(labels.astype(np.int64), minlength=nc)
#         cls_top1_number = np.zeros_like(cls_number)
#         for cls in range(nc):
#             filter_cls = labels == cls
#             top1_number = np.sum(pre[filter_cls] == labels[filter_cls]).tolist()
#             cls_top1_number[cls] = top1_number
#         top1 = (cls_top1_number.sum() / cls_number.sum()).tolist()
#         top1_cls = (cls_top1_number / cls_number).tolist()
#
#         fmt = '<10.3f'
#         space = ' ' * 50
#         LOGGER.info(f'{space}top1: {top1:{fmt}}')
#         return top1, top1_cls, cls_number
#
#     @staticmethod
#     def _get_metrics_stats(predictions, labels, stats):
#         cls_conf, cls_pre = torch.max(predictions, dim=1)
#         stats.append((cls_pre.cpu(), cls_conf.cpu(), labels.cpu()))
