r"""
Meta Valer module for building all valer class.
"""

from utils.mixins import ValDetectMixin

__all__ = ['MetaValDetect']


class MetaValDetect(ValDetectMixin):
    def __init__(self, args=None, last=True,
                 model=None, half=True, dataloader=None, loss_fn=None, cls_names=None, epoch=None, writer=None):
        super(MetaValDetect, self).__init__()
        self.last = last
        self.training = model is not None
        self.time = None
        self.seen = None
        self.writer = writer

    def val_training(self):
        pass

    def val(self):
        pass

    def log_results(self):
        pass
