r"""
SWA module.
"""

import math
import torch

from torch.nn.modules.batchnorm import _BatchNorm

__all__ = ['EMADecayFunction', 'EMABNFunction', 'update_bn']


class EMADecayFunction(object):
    r"""Use in AveragedModel avg_fn=EMADecayFunction()"""

    def __init__(self, decay=0.9999, tau=2000):
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # to more

    @torch.no_grad()
    def __call__(self, averaged_model_parameter, model_parameter, num_averaged):
        r"""d=[0.0, 0.393] corresponds to num_averaged=[0, 1000]"""
        d = self.decay(num_averaged)
        return averaged_model_parameter * d + (1 - d) * model_parameter


class EMABNFunction(object):
    r"""A function of update_bn"""

    def __init__(self, avg_fn=None):
        if avg_fn is None:
            def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter + \
                       (model_parameter - averaged_model_parameter) / (num_averaged + 1)
        self.avg_fn = avg_fn
        self.n_averaged = 0

    @torch.no_grad()
    def update_bn(self, swa_model, model):
        r"""The parameters of them must be corresponding to each other"""
        device = next(swa_model.parameters()).device
        for swa_module, module in zip(swa_model.modules(), model.modules()):
            if isinstance(swa_module, _BatchNorm) and isinstance(module, _BatchNorm):
                if self.n_averaged == 0:
                    swa_module.running_mean = module.running_mean.to(device)
                    swa_module.running_var = module.running_var.to(device)
                    swa_module.num_batches_tracked = module.num_batches_tracked.to(device)
                else:
                    swa_module.running_mean = self.avg_fn(swa_module.running_mean, module.running_mean.to(device),
                                                          self.n_averaged)
                    swa_module.running_var = self.avg_fn(swa_module.running_var, module.running_var.to(device),
                                                         self.n_averaged)
                    swa_module.num_batches_tracked = module.num_batches_tracked.to(device)

        self.n_averaged += 1


@torch.no_grad()
def update_bn(loader, model, device=None, norm=True):
    r"""
    https://arxiv.org/abs/1803.05407
    Updates BatchNorm running_mean, running_var buffers in the model.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for x in loader:
        if isinstance(x, (list, tuple)):
            x = x[0]
        if device is not None:
            if norm:
                x = x.to(device) / 255
            else:
                x = x.to(device)

        model(x)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)
