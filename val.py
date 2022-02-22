r"""
For validating model.
Consist of some Valers.
"""

import torch

from utils import \
    LOGGER, ValMixin

__all__ = ['ValDetect']


class ValDetect(
    ValMixin  # for validating
):
    def __init__(self, args, model=None, half=True, dataloader=None, loss_fn=None):
        super(ValDetect, self).__init__()
        self.training = model is not None

        # val during training
        if self.training:
            self.device = next(model.parameters()).device
            self.half = half
            self.loss_fn = loss_fn
            self.dataloader = dataloader
            if self.half and self.device.type == 'cpu':
                LOGGER.warning(f'The device is {self.device}, half precision only supported on CUDA')
                self.half = False
            self.model = model.half() if self.half else model.float()

    def val(self):
        self.model.eval()
        loss_all, loss_name = self.val_once()
