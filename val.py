r"""
For validating model.
Consist of some Valers.
"""

import torch

from utils import \
    LOGGER, ValDetectMixin

__all__ = ['ValDetectDetect']


class ValDetectDetect(
    ValDetectMixin  # for validating
):
    def __init__(self, args, model=None, half=True, dataloader=None, loss_fn=None):
        super(ValDetectDetect, self).__init__()
        self.training = model is not None

        if self.training:
            # val during training
            self.device = next(model.parameters()).device
            self.half = half
            self.loss_fn = loss_fn
            self.dataloader = dataloader
            if self.half and self.device.type == 'cpu':
                LOGGER.warning(f'The device is {self.device}, half precision only supported on CUDA')
                self.half = False
            self.model = model.half() if self.half else model.float()
        else:
            # val alone
            pass

    def val(self):
        self.model.eval()
        # TODO maybe save something or plot images below
        loss_all, loss_name, stats = self.val_once()
        LOGGER.debug(f'Validating: {loss_name} is {loss_all}')
        ap_all, f1_all, p_all, r_all, cls_name_number = self.compute_metrics(stats)
